"""
Reward Function for Motion Planning

Composite reward function combining:
- Progress rewards
- Safety penalties
- Comfort rewards
- Traffic rule compliance
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward function weights"""

    # Progress rewards
    progress: float = 1.0           # Moving towards goal
    goal_reached: float = 10.0      # Bonus for reaching goal

    # Safety penalties
    collision: float = -10.0        # Collision (terminates episode)
    near_collision: float = -0.5    # TTC < threshold
    off_road: float = -5.0          # Leaving drivable area

    # Comfort penalties
    jerk: float = -0.1              # Sudden acceleration changes
    lateral_acc: float = -0.05      # High lateral acceleration
    steering_rate: float = -0.02    # Rapid steering changes

    # Traffic rule compliance
    lane_keeping: float = 0.5       # Staying in lane
    speed_limit: float = -0.5       # Exceeding speed limit
    traffic_light: float = -5.0     # Running red light

    # Thresholds
    ttc_threshold: float = 2.0      # TTC threshold in seconds
    lane_threshold: float = 0.5     # Lane center offset threshold in meters
    goal_threshold: float = 2.0     # Goal reached distance threshold


@dataclass
class StepInfo:
    """Information about the current step for reward computation"""
    # Position and dynamics
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float

    # Goal information
    goal_position: np.ndarray
    progress: float = 0.0  # Normalized progress towards goal

    # Safety
    collision: bool = False
    ttc: float = float('inf')  # Time-to-collision
    off_road: bool = False

    # Comfort
    prev_acceleration: Optional[np.ndarray] = None
    prev_steering: Optional[float] = None
    lateral_acceleration: float = 0.0

    # Traffic rules
    in_lane: bool = True
    lane_center_offset: float = 0.0
    speed: float = 0.0
    speed_limit: float = 30.0
    traffic_violation: bool = False
    goal_reached: bool = False


class RewardFunction:
    """
    Composite reward function for autonomous driving planning

    Usage:
        config = RewardConfig()
        reward_fn = RewardFunction(config)
        reward, done = reward_fn.compute(state, action, next_state, info)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        info: StepInfo,
        action: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Compute total reward for the current step

        Args:
            info: StepInfo containing state information
            action: Action taken [acceleration, steering]

        Returns:
            reward: Total reward value
            done: Whether episode should terminate
        """
        reward = 0.0
        done = False

        # === Safety (highest priority) ===
        if info.collision:
            reward += self.config.collision
            return reward, True  # Terminate on collision

        if info.ttc < self.config.ttc_threshold:
            # Scale penalty based on how close TTC is to threshold
            ttc_penalty = self.config.near_collision * (
                (self.config.ttc_threshold - info.ttc) / self.config.ttc_threshold
            )
            reward += ttc_penalty

        if info.off_road:
            reward += self.config.off_road

        # === Progress ===
        reward += self.config.progress * info.progress

        if info.goal_reached:
            reward += self.config.goal_reached
            done = True

        # === Comfort ===
        # Jerk (change in acceleration)
        if info.prev_acceleration is not None:
            jerk = np.linalg.norm(info.acceleration - info.prev_acceleration) / 0.1
            reward += self.config.jerk * jerk

        # Lateral acceleration
        reward += self.config.lateral_acc * abs(info.lateral_acceleration)

        # Steering rate
        if info.prev_steering is not None:
            steering_rate = abs(action[1] - info.prev_steering) / 0.1
            reward += self.config.steering_rate * steering_rate

        # === Traffic Rules ===
        # Lane keeping
        if info.in_lane and info.lane_center_offset < self.config.lane_threshold:
            reward += self.config.lane_keeping

        # Speed limit
        if info.speed > info.speed_limit:
            reward += self.config.speed_limit

        # Traffic light violation
        if info.traffic_violation:
            reward += self.config.traffic_light

        return reward, done

    def compute_progress(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        goal_position: np.ndarray,
        max_speed: float = 30.0
    ) -> float:
        """
        Compute progress reward component

        Rewards movement towards the goal.
        """
        # Direction to goal
        goal_direction = goal_position - position
        goal_distance = np.linalg.norm(goal_direction)

        if goal_distance < 1e-6:
            return 1.0

        goal_direction = goal_direction / goal_distance

        # Project velocity onto goal direction
        velocity_towards_goal = np.dot(velocity, goal_direction)

        # Normalize by max speed
        return velocity_towards_goal / max_speed

    def compute_ttc(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        other_position: np.ndarray,
        other_velocity: np.ndarray,
        min_distance: float = 2.0
    ) -> float:
        """
        Compute Time-to-Collision (TTC)

        Returns infinity if no collision is predicted.
        """
        # Relative position and velocity
        rel_position = other_position - ego_position
        rel_velocity = other_velocity - ego_velocity

        # Distance
        distance = np.linalg.norm(rel_position)
        if distance < min_distance:
            return 0.0

        # Closing speed
        closing_speed = -np.dot(rel_position, rel_velocity) / distance

        if closing_speed <= 0:
            return float('inf')  # Moving apart

        # Time to collision
        ttc = (distance - min_distance) / closing_speed

        return max(0.0, ttc)

    def get_reward_breakdown(
        self,
        info: StepInfo,
        action: np.ndarray
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components

        Useful for debugging and analysis.
        """
        breakdown = {}

        # Safety
        breakdown['collision'] = self.config.collision if info.collision else 0.0
        breakdown['near_collision'] = (
            self.config.near_collision * (
                (self.config.ttc_threshold - info.ttc) / self.config.ttc_threshold
            ) if info.ttc < self.config.ttc_threshold else 0.0
        )
        breakdown['off_road'] = self.config.off_road if info.off_road else 0.0

        # Progress
        breakdown['progress'] = self.config.progress * info.progress
        breakdown['goal_reached'] = self.config.goal_reached if info.goal_reached else 0.0

        # Comfort
        if info.prev_acceleration is not None:
            jerk = np.linalg.norm(info.acceleration - info.prev_acceleration) / 0.1
            breakdown['jerk'] = self.config.jerk * jerk
        else:
            breakdown['jerk'] = 0.0

        breakdown['lateral_acc'] = self.config.lateral_acc * abs(info.lateral_acceleration)

        if info.prev_steering is not None:
            steering_rate = abs(action[1] - info.prev_steering) / 0.1
            breakdown['steering_rate'] = self.config.steering_rate * steering_rate
        else:
            breakdown['steering_rate'] = 0.0

        # Traffic rules
        breakdown['lane_keeping'] = (
            self.config.lane_keeping
            if info.in_lane and info.lane_center_offset < self.config.lane_threshold
            else 0.0
        )
        breakdown['speed_limit'] = (
            self.config.speed_limit if info.speed > info.speed_limit else 0.0
        )
        breakdown['traffic_light'] = (
            self.config.traffic_light if info.traffic_violation else 0.0
        )

        breakdown['total'] = sum(breakdown.values())

        return breakdown
