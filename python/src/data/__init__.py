"""
Data Module - Dataset loading and preprocessing

Components:
- loaders: Dataset-specific loaders (nuPlan, Waymo, etc.)
- processors: Data processing pipelines
- augmentation: Data augmentation strategies
- visualization: Data visualization tools
"""

from .base import Scenario, AgentTrack, BaseDataLoader, BaseProcessor
from .processor import PlanningProcessor, PlanningDataset
from .augmentation import AugmentationPipeline
from .splitter import DatasetSplitter
from .visualizer import ScenarioVisualizer


# Lazy imports to avoid import errors when dependencies are not installed
def get_nuplan_loader():
    """Get NuPlan data loader (requires nuplan-devkit)."""
    from .nuplan_loader import NuPlanDataLoader, NuPlanMiniLoader, NuPlanFullLoader
    return NuPlanDataLoader, NuPlanMiniLoader, NuPlanFullLoader


__all__ = [
    # Base classes
    "Scenario",
    "AgentTrack",
    "BaseDataLoader",
    "BaseProcessor",
    # Processing
    "PlanningProcessor",
    "PlanningDataset",
    # Augmentation
    "AugmentationPipeline",
    # Splitting
    "DatasetSplitter",
    # Visualization
    "ScenarioVisualizer",
    # Loader getters
    "get_nuplan_loader",
]
