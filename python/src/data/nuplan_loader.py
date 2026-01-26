"""
nuPlan Dataset Loader
=====================

Loads nuPlan scenarios and converts them to the unified Scenario format.

Requirements:
    pip install nuplan-devkit

Usage:
    from src.data.nuplan_loader import NuPlanDataLoader

    loader = NuPlanDataLoader(
        data_root="~/nuplan/dataset",
        db_files="nuplan-v1.1/mini",
        map_root="maps"
    )

    # Get all scenario IDs
    scenario_ids = loader.get_scenario_ids()

    # Load a single scenario
    scenario = loader.load_scenario(scenario_ids[0])

    # Iterate over all scenarios
    for scenario in loader:
        print(scenario.scenario_id, scenario.duration)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
import numpy as np

from .base import Scenario, AgentTrack, BaseDataLoader

logger = logging.getLogger(__name__)


@dataclass
class NuPlanConfig:
    """Configuration for NuPlan data loading."""
    # Data paths
    data_root: str = os.path.expanduser("~/nuplan/dataset")
    db_files: str = "nuplan-v1.1/mini"
    map_root: str = "maps"

    # Scenario parameters
    scenario_duration: float = 15.0  # seconds
    past_time_horizon: float = 2.0  # seconds of history
    future_time_horizon: float = 8.0  # seconds to predict
    sample_interval: float = 0.1  # 10 Hz

    # Filtering
    scenario_types: Optional[List[str]] = None  # None = all types
    max_scenarios: Optional[int] = None  # Limit for testing

    # Agent filtering
    max_agents: int = 32  # Maximum number of agents per scenario
    min_agent_distance: float = 1.0  # Minimum distance to ego
    max_agent_distance: float = 50.0  # Maximum distance to ego


class NuPlanDataLoader(BaseDataLoader):
    """
    Data loader for nuPlan dataset.

    Converts nuPlan scenarios to the unified Scenario format for training.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        db_files: Optional[str] = None,
        map_root: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize NuPlan data loader.

        Args:
            data_root: Root directory of nuPlan dataset
            db_files: Subdirectory containing .db files
            map_root: Subdirectory containing map data
            config: Additional configuration options
        """
        # Build configuration
        self.nuplan_config = NuPlanConfig()
        if data_root:
            self.nuplan_config.data_root = data_root
        if db_files:
            self.nuplan_config.db_files = db_files
        if map_root:
            self.nuplan_config.map_root = map_root
        if config:
            for key, value in config.items():
                if hasattr(self.nuplan_config, key):
                    setattr(self.nuplan_config, key, value)

        super().__init__(self.nuplan_config.data_root, config)

        # Paths
        self.data_root = Path(self.nuplan_config.data_root).expanduser()
        self.db_path = self.data_root / self.nuplan_config.db_files
        self.map_path = self.data_root / self.nuplan_config.map_root

        # Lazy-loaded components
        self._scenario_builder = None
        self._scenarios = None
        self._scenario_ids = None

        # Validate paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required paths exist."""
        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            logger.info("Run: python scripts/setup_nuplan.py --setup-dirs")

        if not self.db_path.exists():
            logger.warning(f"DB path does not exist: {self.db_path}")
            logger.info("Download nuPlan mini dataset from https://www.nuplan.org/")

        if not self.map_path.exists():
            logger.warning(f"Map path does not exist: {self.map_path}")

    def _init_scenario_builder(self):
        """Initialize nuPlan scenario builder (lazy loading)."""
        if self._scenario_builder is not None:
            return

        try:
            from nuplan.planning.scenario_builder.nuplan_scenario_builder import NuPlanScenarioBuilder
            from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
            from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
        except ImportError:
            raise ImportError(
                "nuplan-devkit not installed. Run:\n"
                "  python scripts/setup_nuplan.py --install-devkit"
            )

        logger.info(f"Initializing NuPlan scenario builder...")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  DB path: {self.db_path}")
        logger.info(f"  Map path: {self.map_path}")

        # Build scenario filter
        scenario_filter = ScenarioFilter(
            scenario_types=self.nuplan_config.scenario_types,
            log_names=None,
            limit_scenarios_per_type=self.nuplan_config.max_scenarios,
        )

        # Initialize scenario builder
        self._scenario_builder = NuPlanScenarioBuilder(
            data_root=str(self.data_root),
            map_root=str(self.map_path),
            db_files=str(self.db_path),
            map_version="nuplan-maps-v1.0",
        )

        # Get scenarios
        self._scenarios = self._scenario_builder.get_scenarios(
            scenario_filter, self._scenario_builder.scenario_mapping
        )

        self._scenario_ids = [s.token for s in self._scenarios]

        logger.info(f"Found {len(self._scenarios)} scenarios")

    def get_scenario_ids(self) -> List[str]:
        """Get list of all available scenario IDs."""
        self._init_scenario_builder()
        return self._scenario_ids.copy()

    def __len__(self) -> int:
        """Return total number of scenarios."""
        self._init_scenario_builder()
        return len(self._scenarios)

    def __iter__(self) -> Iterator[Scenario]:
        """Iterate over all scenarios."""
        self._init_scenario_builder()
        for nuplan_scenario in self._scenarios:
            yield self._convert_scenario(nuplan_scenario)

    def load_scenario(self, scenario_id: str) -> Scenario:
        """
        Load a single scenario by ID.

        Args:
            scenario_id: Scenario token/ID

        Returns:
            Unified Scenario object
        """
        self._init_scenario_builder()

        # Find scenario by token
        nuplan_scenario = None
        for s in self._scenarios:
            if s.token == scenario_id:
                nuplan_scenario = s
                break

        if nuplan_scenario is None:
            raise ValueError(f"Scenario not found: {scenario_id}")

        return self._convert_scenario(nuplan_scenario)

    def _convert_scenario(self, nuplan_scenario) -> Scenario:
        """
        Convert nuPlan scenario to unified Scenario format.

        Args:
            nuplan_scenario: NuPlan AbstractScenario object

        Returns:
            Unified Scenario object
        """
        # Get iteration info
        num_iterations = nuplan_scenario.get_number_of_iterations()
        sample_interval = self.nuplan_config.sample_interval

        # Extract ego trajectory
        ego_trajectory = self._extract_ego_trajectory(nuplan_scenario, num_iterations)

        # Extract agent tracks
        agents = self._extract_agents(nuplan_scenario, num_iterations)

        # Extract map features (optional)
        map_features = self._extract_map_features(nuplan_scenario)

        # Extract traffic lights (optional)
        traffic_lights = self._extract_traffic_lights(nuplan_scenario)

        return Scenario(
            scenario_id=nuplan_scenario.token,
            source="nuplan",
            duration=num_iterations * sample_interval,
            ego_trajectory=ego_trajectory,
            agents=agents,
            map_features=map_features,
            traffic_lights=traffic_lights,
        )

    def _extract_ego_trajectory(
        self,
        nuplan_scenario,
        num_iterations: int
    ) -> np.ndarray:
        """
        Extract ego vehicle trajectory.

        Returns:
            np.ndarray of shape [T, 7]: x, y, vx, vy, ax, ay, heading
        """
        trajectory = []

        for i in range(num_iterations):
            try:
                iteration = nuplan_scenario.get_iteration(i)
                ego_state = iteration.ego_state

                # Position
                x = ego_state.rear_axle.x
                y = ego_state.rear_axle.y

                # Velocity
                vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
                vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y

                # Acceleration
                ax = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
                ay = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y

                # Heading
                heading = ego_state.rear_axle.heading

                trajectory.append([x, y, vx, vy, ax, ay, heading])
            except Exception as e:
                logger.warning(f"Error extracting ego state at iteration {i}: {e}")
                # Use previous state or zeros
                if trajectory:
                    trajectory.append(trajectory[-1])
                else:
                    trajectory.append([0, 0, 0, 0, 0, 0, 0])

        return np.array(trajectory, dtype=np.float32)

    def _extract_agents(
        self,
        nuplan_scenario,
        num_iterations: int
    ) -> List[AgentTrack]:
        """
        Extract other agent tracks.

        Returns:
            List of AgentTrack objects
        """
        # Collect agent data across all timesteps
        agent_data: Dict[str, Dict] = {}

        for i in range(num_iterations):
            try:
                iteration = nuplan_scenario.get_iteration(i)
                detections = iteration.detections

                for detection in detections.tracked_objects:
                    agent_id = detection.track_token

                    if agent_id not in agent_data:
                        agent_data[agent_id] = {
                            "type": self._convert_agent_type(detection.tracked_object_type),
                            "dimensions": (
                                detection.box.length,
                                detection.box.width,
                                detection.box.height,
                            ),
                            "trajectory": {},
                        }

                    # Store state at this timestep
                    agent_data[agent_id]["trajectory"][i] = [
                        detection.center.x,
                        detection.center.y,
                        detection.velocity.x if detection.velocity else 0,
                        detection.velocity.y if detection.velocity else 0,
                        0,  # acceleration not directly available
                        0,
                        detection.center.heading,
                    ]

            except Exception as e:
                logger.warning(f"Error extracting agents at iteration {i}: {e}")

        # Convert to AgentTrack objects
        agents = []
        for agent_id, data in agent_data.items():
            # Build trajectory array
            traj_dict = data["trajectory"]
            traj_array = np.zeros((num_iterations, 7), dtype=np.float32)

            for t, state in traj_dict.items():
                traj_array[t] = state

            # Interpolate missing timesteps
            traj_array = self._interpolate_trajectory(traj_array, list(traj_dict.keys()))

            agents.append(AgentTrack(
                agent_id=agent_id,
                agent_type=data["type"],
                trajectory=traj_array,
                dimensions=data["dimensions"],
            ))

        # Sort by distance to ego and limit
        if agents:
            ego_pos = self._extract_ego_trajectory(nuplan_scenario, 1)[0, :2]
            agents.sort(key=lambda a: np.linalg.norm(a.trajectory[0, :2] - ego_pos))
            agents = agents[:self.nuplan_config.max_agents]

        return agents

    def _convert_agent_type(self, nuplan_type) -> str:
        """Convert nuPlan tracked object type to string."""
        type_map = {
            0: "vehicle",
            1: "pedestrian",
            2: "bicycle",
            3: "traffic_cone",
            4: "barrier",
            5: "czone_sign",
            6: "generic_object",
        }
        if hasattr(nuplan_type, 'value'):
            return type_map.get(nuplan_type.value, "unknown")
        return type_map.get(int(nuplan_type), "unknown")

    def _interpolate_trajectory(
        self,
        trajectory: np.ndarray,
        valid_indices: List[int]
    ) -> np.ndarray:
        """Interpolate missing timesteps in trajectory."""
        if len(valid_indices) <= 1:
            return trajectory

        valid_indices = sorted(valid_indices)

        for dim in range(trajectory.shape[1]):
            valid_values = trajectory[valid_indices, dim]
            # Simple linear interpolation
            trajectory[:, dim] = np.interp(
                np.arange(trajectory.shape[0]),
                valid_indices,
                valid_values
            )

        return trajectory

    def _extract_map_features(self, nuplan_scenario) -> Optional[Dict[str, Any]]:
        """Extract map features for the scenario."""
        try:
            map_api = nuplan_scenario.map_api

            # Get ego position for local map extraction
            ego_state = nuplan_scenario.initial_ego_state
            ego_pos = (ego_state.rear_axle.x, ego_state.rear_axle.y)

            # Extract nearby lanes
            radius = 50.0  # meters
            nearby_lanes = map_api.get_proximal_map_objects(
                ego_pos, radius, ["lane", "lane_connector"]
            )

            return {
                "map_name": nuplan_scenario.map_api.map_name,
                "ego_position": ego_pos,
                "nearby_lane_count": len(nearby_lanes.get("lane", [])),
            }
        except Exception as e:
            logger.warning(f"Error extracting map features: {e}")
            return None

    def _extract_traffic_lights(self, nuplan_scenario) -> Optional[List[Dict]]:
        """Extract traffic light states."""
        try:
            traffic_lights = []
            iteration = nuplan_scenario.get_iteration(0)

            for tl in iteration.traffic_light_status:
                traffic_lights.append({
                    "lane_connector_id": tl.lane_connector_id,
                    "status": str(tl.status),
                })

            return traffic_lights if traffic_lights else None
        except Exception as e:
            logger.warning(f"Error extracting traffic lights: {e}")
            return None


class NuPlanMiniLoader(NuPlanDataLoader):
    """Convenience loader for nuPlan Mini dataset."""

    def __init__(self, data_root: Optional[str] = None, **kwargs):
        super().__init__(
            data_root=data_root,
            db_files="nuplan-v1.1/mini",
            **kwargs
        )


class NuPlanFullLoader(NuPlanDataLoader):
    """Convenience loader for full nuPlan dataset."""

    def __init__(self, data_root: Optional[str] = None, split: str = "trainval", **kwargs):
        super().__init__(
            data_root=data_root,
            db_files=f"nuplan-v1.1/{split}",
            **kwargs
        )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing NuPlan Data Loader...")
    print("=" * 60)

    try:
        loader = NuPlanMiniLoader()
        print(f"Data root: {loader.data_root}")
        print(f"DB path: {loader.db_path}")
        print(f"Map path: {loader.map_path}")

        # Try to load scenarios
        scenario_ids = loader.get_scenario_ids()
        print(f"\nFound {len(scenario_ids)} scenarios")

        if scenario_ids:
            print(f"\nLoading first scenario: {scenario_ids[0]}")
            scenario = loader.load_scenario(scenario_ids[0])
            print(f"  Duration: {scenario.duration:.2f}s")
            print(f"  Ego trajectory shape: {scenario.ego_trajectory.shape}")
            print(f"  Number of agents: {len(scenario.agents)}")
            if scenario.map_features:
                print(f"  Map: {scenario.map_features.get('map_name')}")

    except FileNotFoundError as e:
        print(f"\nDataset not found: {e}")
        print("\nTo set up nuPlan:")
        print("1. Run: python scripts/setup_nuplan.py --setup-dirs")
        print("2. Download from: https://www.nuplan.org/")
        print("3. Run: python scripts/setup_nuplan.py --verify")

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nTo install nuplan-devkit:")
        print("  python scripts/setup_nuplan.py --install-devkit")
