"""
Scenario Processor
==================

Converts raw Scenario objects into training-ready tensors for Planning models.

Pipeline:
    Scenario → Normalize → Window → Features → (obs, action) pairs

Usage:
    from src.data.processor import PlanningProcessor

    processor = PlanningProcessor(config={
        "history_steps": 20,      # 2 seconds at 10Hz
        "future_steps": 80,       # 8 seconds at 10Hz
        "max_agents": 32,
    })

    # Process single scenario
    samples = processor.process(scenario)

    # Create PyTorch dataset
    dataset = processor.create_dataset(scenarios)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from .base import Scenario, AgentTrack, BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for scenario processing."""
    # Time parameters (at 10Hz)
    history_steps: int = 20          # 2.0 seconds of history
    future_steps: int = 80           # 8.0 seconds of future
    sample_interval: float = 0.1     # 10 Hz

    # Agent parameters
    max_agents: int = 32             # Maximum agents per sample
    agent_features: int = 7          # x, y, vx, vy, ax, ay, heading
    min_agent_history: int = 5       # Minimum history for valid agent

    # Ego parameters
    ego_features: int = 7            # x, y, vx, vy, ax, ay, heading

    # Normalization
    normalize: bool = True
    position_scale: float = 50.0     # meters
    velocity_scale: float = 20.0     # m/s
    acceleration_scale: float = 5.0  # m/s²
    heading_scale: float = np.pi     # radians

    # Action space
    action_dim: int = 2              # acceleration, steering
    max_acceleration: float = 4.0    # m/s² (absolute)
    max_steering: float = 0.5        # rad

    # Output format
    use_relative_coords: bool = True  # Relative to ego at current timestep
    include_map: bool = False         # Include map features
    include_traffic_lights: bool = False


class PlanningProcessor(BaseProcessor):
    """
    Processes scenarios into training samples for Planning models.

    Each sample contains:
    - observation: ego history + agent states + route info
    - action: future ego acceleration and steering
    - future_trajectory: ground truth future path (for IL)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        self.cfg = ProcessorConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)

    def process(self, scenario: Scenario) -> List[Dict[str, np.ndarray]]:
        """
        Process a scenario into training samples.

        Slides a window across the scenario to generate multiple samples.

        Args:
            scenario: Unified Scenario object

        Returns:
            List of sample dicts with 'observation', 'action', 'future_trajectory'
        """
        samples = []
        total_steps = scenario.ego_trajectory.shape[0]
        window = self.cfg.history_steps + self.cfg.future_steps

        if total_steps < window:
            logger.warning(
                f"Scenario {scenario.scenario_id} too short: "
                f"{total_steps} < {window} steps"
            )
            return samples

        # Slide window
        stride = max(1, self.cfg.future_steps // 4)  # 25% overlap
        for t in range(self.cfg.history_steps, total_steps - self.cfg.future_steps, stride):
            try:
                sample = self._create_sample(scenario, t)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Error at timestep {t}: {e}")

        return samples

    def _create_sample(self, scenario: Scenario, t: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Create a single training sample at timestep t.

        Args:
            scenario: Source scenario
            t: Current timestep (split between history and future)

        Returns:
            Dict with observation, action, future_trajectory, metadata
        """
        # Extract time ranges
        hist_start = t - self.cfg.history_steps
        hist_end = t
        future_end = t + self.cfg.future_steps

        # Ego history and future
        ego_history = scenario.ego_trajectory[hist_start:hist_end].copy()
        ego_future = scenario.ego_trajectory[hist_end:future_end].copy()

        # Current ego state (reference frame)
        ego_current = scenario.ego_trajectory[t].copy()

        # Transform to ego-relative coordinates
        if self.cfg.use_relative_coords:
            ego_history = self._to_ego_relative(ego_history, ego_current)
            ego_future = self._to_ego_relative(ego_future, ego_current)

        # Extract agent states
        agent_states = self._extract_agent_states(scenario.agents, t, ego_current)

        # Create observation
        observation = self.create_observation(scenario, t)

        # Create action (from future trajectory)
        action = self.create_action(scenario, t)

        # Normalize
        if self.cfg.normalize:
            observation = self._normalize_observation(observation)
            ego_future = self._normalize_trajectory(ego_future)

        return {
            "observation": observation.astype(np.float32),
            "action": action.astype(np.float32),
            "future_trajectory": ego_future[:, :2].astype(np.float32),  # x, y only
            "ego_history": ego_history.astype(np.float32),
            "agent_states": agent_states.astype(np.float32),
            "metadata": {
                "scenario_id": scenario.scenario_id,
                "timestep": t,
                "ego_speed": float(np.sqrt(ego_current[2]**2 + ego_current[3]**2)),
            }
        }

    def create_observation(self, scenario: Scenario, timestep: int) -> np.ndarray:
        """
        Create observation vector for Planning model.

        Structure (~140D):
        - ego_state: [8D] position, velocity, acceleration, heading, speed
        - ego_history: [history_steps * 2D] past x,y positions
        - agent_states: [max_agents * 5D] relative pos, vel, heading
        - route_info: [30D] future waypoints (from ego trajectory)

        Args:
            scenario: Source scenario
            timestep: Current timestep

        Returns:
            1D observation vector
        """
        ego_traj = scenario.ego_trajectory
        ego_current = ego_traj[timestep]

        # 1. Ego state [8D]
        speed = np.sqrt(ego_current[2]**2 + ego_current[3]**2)
        ego_state = np.array([
            ego_current[0],  # x
            ego_current[1],  # y
            ego_current[2],  # vx
            ego_current[3],  # vy
            ego_current[4],  # ax
            ego_current[5],  # ay
            ego_current[6],  # heading
            speed,           # speed magnitude
        ])

        # 2. Ego history [history_steps * 2D = 40D]
        hist_start = max(0, timestep - self.cfg.history_steps)
        ego_history = ego_traj[hist_start:timestep, :2].copy()  # x, y only

        if self.cfg.use_relative_coords:
            ego_history = ego_history - ego_current[:2]
            # Rotate to ego heading
            ego_history = self._rotate_points(ego_history, -ego_current[6])

        # Pad if needed
        if ego_history.shape[0] < self.cfg.history_steps:
            pad = np.zeros((self.cfg.history_steps - ego_history.shape[0], 2))
            ego_history = np.vstack([pad, ego_history])

        ego_history_flat = ego_history.flatten()  # [history_steps * 2]

        # 3. Agent states [max_agents * 5D = 160D]
        agent_states = self._extract_agent_features(
            scenario.agents, timestep, ego_current
        )
        agent_flat = agent_states.flatten()  # [max_agents * 5]

        # 4. Route info [30D] - future waypoints as proxy for route
        future_end = min(timestep + 30, ego_traj.shape[0])
        route = ego_traj[timestep:future_end, :2].copy()

        if self.cfg.use_relative_coords:
            route = route - ego_current[:2]
            route = self._rotate_points(route, -ego_current[6])

        # Pad/truncate to fixed size
        if route.shape[0] < 15:
            pad = np.zeros((15 - route.shape[0], 2))
            route = np.vstack([route, pad])
        else:
            route = route[:15]

        route_flat = route.flatten()  # [30D]

        # Concatenate all features
        observation = np.concatenate([
            ego_state,          # 8D
            ego_history_flat,   # 40D
            agent_flat,         # 160D
            route_flat,         # 30D
        ])  # Total: ~238D

        return observation

    def create_action(self, scenario: Scenario, timestep: int) -> np.ndarray:
        """
        Create action vector from future trajectory.

        Computes acceleration and steering from trajectory difference.

        Args:
            scenario: Source scenario
            timestep: Current timestep

        Returns:
            Action vector [acceleration, steering]
        """
        ego_traj = scenario.ego_trajectory
        dt = self.cfg.sample_interval

        # Current and next state
        current = ego_traj[timestep]
        if timestep + 1 < ego_traj.shape[0]:
            next_state = ego_traj[timestep + 1]
        else:
            next_state = current

        # Acceleration: change in speed
        speed_current = np.sqrt(current[2]**2 + current[3]**2)
        speed_next = np.sqrt(next_state[2]**2 + next_state[3]**2)
        acceleration = (speed_next - speed_current) / dt

        # Steering: change in heading
        heading_diff = next_state[6] - current[6]
        # Normalize to [-pi, pi]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        steering = heading_diff / dt  # heading rate as proxy for steering

        # Clamp to valid range
        acceleration = np.clip(acceleration, -self.cfg.max_acceleration, self.cfg.max_acceleration)
        steering = np.clip(steering, -self.cfg.max_steering, self.cfg.max_steering)

        return np.array([acceleration, steering], dtype=np.float32)

    def _extract_agent_states(
        self,
        agents: List[AgentTrack],
        timestep: int,
        ego_state: np.ndarray
    ) -> np.ndarray:
        """Extract agent states relative to ego."""
        result = np.zeros((self.cfg.max_agents, self.cfg.agent_features), dtype=np.float32)

        for i, agent in enumerate(agents[:self.cfg.max_agents]):
            if timestep < agent.trajectory.shape[0]:
                agent_state = agent.trajectory[timestep].copy()

                if self.cfg.use_relative_coords:
                    # Relative position
                    agent_state[:2] -= ego_state[:2]
                    agent_state[:2] = self._rotate_point(agent_state[:2], -ego_state[6])
                    # Relative heading
                    agent_state[6] -= ego_state[6]

                result[i] = agent_state

        return result

    def _extract_agent_features(
        self,
        agents: List[AgentTrack],
        timestep: int,
        ego_state: np.ndarray
    ) -> np.ndarray:
        """
        Extract compact agent features [max_agents, 5].
        Features: rel_x, rel_y, rel_vx, rel_vy, rel_heading
        """
        result = np.zeros((self.cfg.max_agents, 5), dtype=np.float32)

        for i, agent in enumerate(agents[:self.cfg.max_agents]):
            if timestep >= agent.trajectory.shape[0]:
                continue

            agent_state = agent.trajectory[timestep]

            # Relative position
            rel_pos = agent_state[:2] - ego_state[:2]
            if self.cfg.use_relative_coords:
                rel_pos = self._rotate_point(rel_pos, -ego_state[6])

            # Relative velocity
            rel_vel = agent_state[2:4] - ego_state[2:4]

            # Relative heading
            rel_heading = agent_state[6] - ego_state[6]
            rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

            result[i] = [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], rel_heading]

        return result

    def _to_ego_relative(self, trajectory: np.ndarray, ego_ref: np.ndarray) -> np.ndarray:
        """Transform trajectory to ego-relative coordinates."""
        result = trajectory.copy()

        # Position: translate then rotate
        result[:, :2] -= ego_ref[:2]
        result[:, :2] = self._rotate_points(result[:, :2], -ego_ref[6])

        # Velocity: rotate only
        result[:, 2:4] = self._rotate_points(result[:, 2:4], -ego_ref[6])

        # Acceleration: rotate only
        result[:, 4:6] = self._rotate_points(result[:, 4:6], -ego_ref[6])

        # Heading: relative
        result[:, 6] -= ego_ref[6]
        result[:, 6] = (result[:, 6] + np.pi) % (2 * np.pi) - np.pi

        return result

    def _rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        """Rotate 2D points by angle."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return points @ rot.T

    def _rotate_point(self, point: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a single 2D point."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x = point[0] * cos_a - point[1] * sin_a
        y = point[0] * sin_a + point[1] * cos_a
        return np.array([x, y])

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation features."""
        result = obs.copy()

        # Ego state [8D]: pos, vel, acc, heading, speed
        result[0:2] /= self.cfg.position_scale
        result[2:4] /= self.cfg.velocity_scale
        result[4:6] /= self.cfg.acceleration_scale
        result[6] /= self.cfg.heading_scale
        result[7] /= self.cfg.velocity_scale

        # Ego history [40D]: positions
        history_end = 8 + self.cfg.history_steps * 2
        result[8:history_end] /= self.cfg.position_scale

        # Agent states [160D]: pos, vel, heading
        agent_start = history_end
        agent_end = agent_start + self.cfg.max_agents * 5
        for i in range(self.cfg.max_agents):
            offset = agent_start + i * 5
            result[offset:offset+2] /= self.cfg.position_scale    # position
            result[offset+2:offset+4] /= self.cfg.velocity_scale  # velocity
            result[offset+4] /= self.cfg.heading_scale             # heading

        # Route [30D]: positions
        result[agent_end:] /= self.cfg.position_scale

        return result

    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Normalize trajectory."""
        result = trajectory.copy()
        result[:, :2] /= self.cfg.position_scale      # position
        result[:, 2:4] /= self.cfg.velocity_scale     # velocity
        result[:, 4:6] /= self.cfg.acceleration_scale # acceleration
        result[:, 6] /= self.cfg.heading_scale        # heading
        return result

    def create_dataset(
        self,
        scenarios: List[Scenario],
        max_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create a complete dataset from multiple scenarios.

        Args:
            scenarios: List of Scenario objects
            max_samples: Maximum number of samples to generate

        Returns:
            Dict with stacked arrays: observations, actions, future_trajectories
        """
        all_observations = []
        all_actions = []
        all_futures = []

        for scenario in scenarios:
            samples = self.process(scenario)
            for sample in samples:
                all_observations.append(sample["observation"])
                all_actions.append(sample["action"])
                all_futures.append(sample["future_trajectory"])

                if max_samples and len(all_observations) >= max_samples:
                    break

            if max_samples and len(all_observations) >= max_samples:
                break

        if not all_observations:
            raise ValueError("No valid samples generated from scenarios")

        return {
            "observations": np.stack(all_observations),
            "actions": np.stack(all_actions),
            "future_trajectories": np.stack(all_futures),
            "num_samples": len(all_observations),
        }


class PlanningDataset:
    """
    PyTorch-compatible dataset wrapper.

    Usage:
        dataset = PlanningDataset(processor, scenarios)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(
        self,
        processor: PlanningProcessor,
        scenarios: List[Scenario],
        max_samples: Optional[int] = None
    ):
        self.data = processor.create_dataset(scenarios, max_samples)
        logger.info(f"Created dataset with {self.data['num_samples']} samples")

    def __len__(self) -> int:
        return self.data["num_samples"]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data["observations"][idx],
            self.data["actions"][idx],
            self.data["future_trajectories"][idx],
        )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create a dummy scenario for testing
    T = 200  # 20 seconds at 10Hz
    ego_traj = np.zeros((T, 7), dtype=np.float32)
    ego_traj[:, 0] = np.linspace(0, 100, T)  # x: 0 to 100m
    ego_traj[:, 2] = 5.0  # vx = 5 m/s
    ego_traj[:, 6] = 0.0  # heading = 0

    scenario = Scenario(
        scenario_id="test_001",
        source="test",
        duration=20.0,
        ego_trajectory=ego_traj,
        agents=[],
    )

    processor = PlanningProcessor()
    samples = processor.process(scenario)

    print(f"Generated {len(samples)} samples from scenario")
    if samples:
        s = samples[0]
        print(f"  Observation shape: {s['observation'].shape}")
        print(f"  Action shape: {s['action'].shape}")
        print(f"  Future trajectory shape: {s['future_trajectory'].shape}")
        print(f"  Action values: acc={s['action'][0]:.3f}, steer={s['action'][1]:.3f}")
