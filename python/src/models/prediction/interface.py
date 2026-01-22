"""
Prediction output interface for Planning module
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class PredictedTrajectory:
    """Single agent's predicted future trajectory"""
    agent_id: int
    trajectory: np.ndarray  # [T, 2] future (x, y) positions
    timestamps: np.ndarray  # [T] time from current (in seconds)
    confidence: float  # prediction confidence


@dataclass
class PredictionOutput:
    """Prediction module output to Planning"""
    current_time: float
    horizon: float  # prediction horizon in seconds
    predictions: Dict[int, List[PredictedTrajectory]]  # agent_id -> trajectories


def constant_velocity_prediction(
    position: np.ndarray,  # [x, y]
    velocity: np.ndarray,  # [vx, vy]
    horizon: float = 5.0,
    dt: float = 0.1
) -> np.ndarray:
    """
    Simple constant velocity prediction baseline

    Args:
        position: Current [x, y] position
        velocity: Current [vx, vy] velocity
        horizon: Prediction horizon in seconds
        dt: Time step in seconds

    Returns:
        np.ndarray of shape [T, 2] with predicted positions
    """
    n_steps = int(horizon / dt)
    trajectory = np.zeros((n_steps, 2), dtype=np.float32)

    for t in range(n_steps):
        time = (t + 1) * dt
        trajectory[t, 0] = position[0] + velocity[0] * time
        trajectory[t, 1] = position[1] + velocity[1] * time

    return trajectory


def predictions_to_observation(
    output: PredictionOutput,
    max_agents: int = 8,
    n_future_steps: int = 10
) -> np.ndarray:
    """
    Convert prediction output to fixed-size observation vector

    Args:
        output: PredictionOutput from prediction module
        max_agents: Maximum number of agents to include
        n_future_steps: Number of future timesteps to include

    Returns:
        np.ndarray of shape [max_agents * n_future_steps * 2]
    """
    obs = np.zeros((max_agents, n_future_steps, 2), dtype=np.float32)

    for i, (agent_id, trajectories) in enumerate(output.predictions.items()):
        if i >= max_agents:
            break

        # Use highest confidence trajectory
        best_traj = max(trajectories, key=lambda t: t.confidence)
        traj_len = min(len(best_traj.trajectory), n_future_steps)
        obs[i, :traj_len] = best_traj.trajectory[:traj_len]

    return obs.flatten()
