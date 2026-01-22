"""
Perception output interface for Planning module
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DetectedObject:
    """Single detected object for Planning input"""
    object_id: int
    object_class: str  # vehicle, pedestrian, cyclist
    position: np.ndarray  # [x, y, z] in ego frame
    velocity: np.ndarray  # [vx, vy, vz]
    dimensions: np.ndarray  # [length, width, height]
    heading: float  # radians
    confidence: float  # 0.0 - 1.0


@dataclass
class PerceptionOutput:
    """Perception module output to Planning"""
    timestamp: float
    objects: List[DetectedObject]
    ego_position: np.ndarray  # [x, y, z]
    ego_velocity: np.ndarray  # [vx, vy, vz]
    ego_heading: float
    bev_features: Optional[np.ndarray] = None  # [H, W, C] optional BEV encoding


def objects_to_observation(
    output: PerceptionOutput,
    max_objects: int = 8,
    feature_dim: int = 5
) -> np.ndarray:
    """
    Convert perception output to fixed-size observation vector

    Args:
        output: PerceptionOutput from perception module
        max_objects: Maximum number of objects to include
        feature_dim: Features per object (x, y, vx, vy, heading)

    Returns:
        np.ndarray of shape [max_objects * feature_dim]
    """
    obs = np.zeros((max_objects, feature_dim), dtype=np.float32)

    # Sort objects by distance to ego
    sorted_objects = sorted(
        output.objects,
        key=lambda o: np.linalg.norm(o.position[:2])
    )[:max_objects]

    for i, obj in enumerate(sorted_objects):
        obs[i, 0] = obj.position[0]  # x (relative to ego)
        obs[i, 1] = obj.position[1]  # y
        obs[i, 2] = obj.velocity[0]  # vx
        obs[i, 3] = obj.velocity[1]  # vy
        obs[i, 4] = obj.heading      # heading

    return obs.flatten()
