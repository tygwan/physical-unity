"""
Data Augmentation Module
=========================

Augmentation strategies for driving scenarios to improve model generalization.

Supported augmentations:
- Trajectory noise injection
- Random rotation (heading perturbation)
- Agent dropout (remove random agents)
- Speed scaling
- Mirror (left-right flip)
- Time crop (random window)

Usage:
    from src.data.augmentation import AugmentationPipeline, TrajectoryNoise, MirrorFlip

    augmentor = AugmentationPipeline([
        TrajectoryNoise(position_std=0.1, velocity_std=0.05),
        MirrorFlip(probability=0.5),
        AgentDropout(drop_rate=0.1),
    ])

    augmented_scenario = augmentor(scenario)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from copy import deepcopy

from .base import Scenario, AgentTrack

logger = logging.getLogger(__name__)


class BaseAugmentation:
    """Base class for augmentations."""

    def __init__(self, probability: float = 1.0):
        self.probability = probability

    def __call__(self, scenario: Scenario) -> Scenario:
        if np.random.random() < self.probability:
            return self.apply(scenario)
        return scenario

    def apply(self, scenario: Scenario) -> Scenario:
        raise NotImplementedError


class TrajectoryNoise(BaseAugmentation):
    """Add Gaussian noise to trajectories."""

    def __init__(
        self,
        position_std: float = 0.1,    # meters
        velocity_std: float = 0.05,   # m/s
        heading_std: float = 0.01,    # radians
        probability: float = 0.8,
    ):
        super().__init__(probability)
        self.position_std = position_std
        self.velocity_std = velocity_std
        self.heading_std = heading_std

    def apply(self, scenario: Scenario) -> Scenario:
        scenario = deepcopy(scenario)
        T = scenario.ego_trajectory.shape[0]

        # Add noise to ego trajectory
        noise = np.zeros_like(scenario.ego_trajectory)
        noise[:, :2] = np.random.normal(0, self.position_std, (T, 2))
        noise[:, 2:4] = np.random.normal(0, self.velocity_std, (T, 2))
        noise[:, 6] = np.random.normal(0, self.heading_std, T)

        scenario.ego_trajectory += noise

        # Add noise to agent trajectories
        for agent in scenario.agents:
            T_a = agent.trajectory.shape[0]
            agent_noise = np.zeros_like(agent.trajectory)
            agent_noise[:, :2] = np.random.normal(0, self.position_std, (T_a, 2))
            agent_noise[:, 2:4] = np.random.normal(0, self.velocity_std, (T_a, 2))
            agent_noise[:, 6] = np.random.normal(0, self.heading_std, T_a)
            agent.trajectory += agent_noise

        return scenario


class MirrorFlip(BaseAugmentation):
    """Mirror scenario left-right (flip y-axis and heading)."""

    def __init__(self, probability: float = 0.5):
        super().__init__(probability)

    def apply(self, scenario: Scenario) -> Scenario:
        scenario = deepcopy(scenario)

        # Flip ego: y → -y, vy → -vy, ay → -ay, heading → -heading
        scenario.ego_trajectory[:, 1] *= -1   # y
        scenario.ego_trajectory[:, 3] *= -1   # vy
        scenario.ego_trajectory[:, 5] *= -1   # ay
        scenario.ego_trajectory[:, 6] *= -1   # heading

        # Flip agents
        for agent in scenario.agents:
            agent.trajectory[:, 1] *= -1
            agent.trajectory[:, 3] *= -1
            agent.trajectory[:, 5] *= -1
            agent.trajectory[:, 6] *= -1

        return scenario


class AgentDropout(BaseAugmentation):
    """Randomly remove agents from scenario."""

    def __init__(self, drop_rate: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        self.drop_rate = drop_rate

    def apply(self, scenario: Scenario) -> Scenario:
        scenario = deepcopy(scenario)

        if not scenario.agents:
            return scenario

        # Keep each agent with probability (1 - drop_rate)
        keep_mask = np.random.random(len(scenario.agents)) > self.drop_rate
        scenario.agents = [a for a, keep in zip(scenario.agents, keep_mask) if keep]

        return scenario


class SpeedScaling(BaseAugmentation):
    """Scale all velocities by a random factor."""

    def __init__(
        self,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        probability: float = 0.5,
    ):
        super().__init__(probability)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def apply(self, scenario: Scenario) -> Scenario:
        scenario = deepcopy(scenario)
        scale = np.random.uniform(self.min_scale, self.max_scale)

        # Scale ego velocity and acceleration
        scenario.ego_trajectory[:, 2:4] *= scale   # velocity
        scenario.ego_trajectory[:, 4:6] *= scale   # acceleration

        # Scale agent velocities
        for agent in scenario.agents:
            agent.trajectory[:, 2:4] *= scale
            agent.trajectory[:, 4:6] *= scale

        return scenario


class RandomRotation(BaseAugmentation):
    """Rotate entire scenario by a random angle."""

    def __init__(
        self,
        max_angle: float = 0.1,  # radians (~5.7 degrees)
        probability: float = 0.3,
    ):
        super().__init__(probability)
        self.max_angle = max_angle

    def apply(self, scenario: Scenario) -> Scenario:
        scenario = deepcopy(scenario)
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotate ego trajectory
        self._rotate_trajectory(scenario.ego_trajectory, cos_a, sin_a, angle)

        # Rotate agents
        for agent in scenario.agents:
            self._rotate_trajectory(agent.trajectory, cos_a, sin_a, angle)

        return scenario

    def _rotate_trajectory(self, traj: np.ndarray, cos_a: float, sin_a: float, angle: float):
        """Rotate trajectory in-place."""
        # Position
        x, y = traj[:, 0].copy(), traj[:, 1].copy()
        traj[:, 0] = x * cos_a - y * sin_a
        traj[:, 1] = x * sin_a + y * cos_a

        # Velocity
        vx, vy = traj[:, 2].copy(), traj[:, 3].copy()
        traj[:, 2] = vx * cos_a - vy * sin_a
        traj[:, 3] = vx * sin_a + vy * cos_a

        # Acceleration
        ax, ay = traj[:, 4].copy(), traj[:, 5].copy()
        traj[:, 4] = ax * cos_a - ay * sin_a
        traj[:, 5] = ax * sin_a + ay * cos_a

        # Heading
        traj[:, 6] += angle


class TimeCrop(BaseAugmentation):
    """Randomly crop a time window from the scenario."""

    def __init__(
        self,
        min_duration: float = 10.0,  # seconds
        probability: float = 0.3,
    ):
        super().__init__(probability)
        self.min_steps = int(min_duration / 0.1)  # at 10Hz

    def apply(self, scenario: Scenario) -> Scenario:
        T = scenario.ego_trajectory.shape[0]

        if T <= self.min_steps:
            return scenario

        scenario = deepcopy(scenario)

        # Random window
        crop_length = np.random.randint(self.min_steps, T + 1)
        start = np.random.randint(0, T - crop_length + 1)
        end = start + crop_length

        # Crop ego
        scenario.ego_trajectory = scenario.ego_trajectory[start:end]
        scenario.duration = crop_length * 0.1

        # Crop agents
        for agent in scenario.agents:
            if agent.trajectory.shape[0] >= end:
                agent.trajectory = agent.trajectory[start:end]
            else:
                # Pad with last known state
                padded = np.zeros((crop_length, 7), dtype=np.float32)
                valid_end = min(agent.trajectory.shape[0], end) - start
                padded[:valid_end] = agent.trajectory[start:start + valid_end]
                if valid_end > 0:
                    padded[valid_end:] = padded[valid_end - 1]
                agent.trajectory = padded

        return scenario


class AugmentationPipeline:
    """
    Sequential augmentation pipeline.

    Applies augmentations in order, each with its own probability.
    """

    def __init__(self, augmentations: Optional[List[BaseAugmentation]] = None):
        if augmentations is None:
            # Default augmentation set
            augmentations = [
                TrajectoryNoise(position_std=0.1, velocity_std=0.05, probability=0.8),
                MirrorFlip(probability=0.5),
                AgentDropout(drop_rate=0.1, probability=0.3),
                SpeedScaling(min_scale=0.9, max_scale=1.1, probability=0.3),
                RandomRotation(max_angle=0.05, probability=0.2),
            ]

        self.augmentations = augmentations

    def __call__(self, scenario: Scenario) -> Scenario:
        for aug in self.augmentations:
            scenario = aug(scenario)
        return scenario

    def augment_batch(
        self,
        scenarios: List[Scenario],
        num_augmented: int = 1
    ) -> List[Scenario]:
        """
        Augment a batch of scenarios.

        Args:
            scenarios: Original scenarios
            num_augmented: Number of augmented copies per scenario

        Returns:
            Original + augmented scenarios
        """
        result = list(scenarios)  # Keep originals

        for scenario in scenarios:
            for _ in range(num_augmented):
                result.append(self(scenario))

        return result


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy scenario
    T = 150
    ego = np.zeros((T, 7), dtype=np.float32)
    ego[:, 0] = np.linspace(0, 75, T)  # x
    ego[:, 2] = 5.0  # vx
    ego[:, 6] = 0.1  # slight heading

    agent_traj = np.zeros((T, 7), dtype=np.float32)
    agent_traj[:, 0] = np.linspace(5, 80, T)
    agent_traj[:, 1] = 3.5  # lane offset
    agent_traj[:, 2] = 5.5

    scenario = Scenario(
        scenario_id="aug_test",
        source="test",
        duration=15.0,
        ego_trajectory=ego,
        agents=[AgentTrack("a1", "vehicle", agent_traj, (4.5, 1.8, 1.5))],
    )

    # Test pipeline
    pipeline = AugmentationPipeline()

    print("Original:")
    print(f"  Ego start: ({scenario.ego_trajectory[0, 0]:.2f}, {scenario.ego_trajectory[0, 1]:.2f})")
    print(f"  Agent count: {len(scenario.agents)}")

    augmented = pipeline(scenario)
    print("\nAugmented:")
    print(f"  Ego start: ({augmented.ego_trajectory[0, 0]:.2f}, {augmented.ego_trajectory[0, 1]:.2f})")
    print(f"  Agent count: {len(augmented.agents)}")

    # Batch augmentation
    batch = pipeline.augment_batch([scenario], num_augmented=3)
    print(f"\nBatch: {len(batch)} scenarios (1 original + 3 augmented)")
