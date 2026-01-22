"""
Base classes for data loading and processing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class Scenario:
    """Unified scenario representation across datasets"""
    scenario_id: str
    source: str  # nuplan, waymo, highd, simulation
    duration: float  # seconds

    # Ego vehicle trajectory: [T, 7] (x, y, vx, vy, ax, ay, heading)
    ego_trajectory: np.ndarray

    # Other agents
    agents: List["AgentTrack"]

    # Map information (optional)
    map_features: Optional[Dict[str, Any]] = None

    # Traffic lights (optional)
    traffic_lights: Optional[List[Dict]] = None


@dataclass
class AgentTrack:
    """Single agent track over time"""
    agent_id: str
    agent_type: str  # vehicle, pedestrian, cyclist
    trajectory: np.ndarray  # [T, 7] (x, y, vx, vy, ax, ay, heading)
    dimensions: Tuple[float, float, float]  # length, width, height


class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders"""

    def __init__(self, data_path: str, config: Optional[Dict] = None):
        self.data_path = data_path
        self.config = config or {}

    @abstractmethod
    def load_scenario(self, scenario_id: str) -> Scenario:
        """Load a single scenario by ID"""
        pass

    @abstractmethod
    def get_scenario_ids(self) -> List[str]:
        """Get list of all available scenario IDs"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of scenarios"""
        pass

    def __iter__(self):
        """Iterate over all scenarios"""
        for scenario_id in self.get_scenario_ids():
            yield self.load_scenario(scenario_id)


class BaseProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def process(self, scenario: Scenario) -> Dict[str, Any]:
        """Process a scenario into training-ready format"""
        pass

    @abstractmethod
    def create_observation(self, scenario: Scenario, timestep: int) -> np.ndarray:
        """Create observation vector for a given timestep"""
        pass

    @abstractmethod
    def create_action(self, scenario: Scenario, timestep: int) -> np.ndarray:
        """Create action vector for a given timestep"""
        pass
