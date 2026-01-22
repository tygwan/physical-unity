"""
Evaluation Metrics for Autonomous Driving

Safety, Progress, Comfort, and Efficiency metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode"""
    episode_id: int
    total_reward: float
    steps: int
    collision: bool
    goal_reached: bool
    distance_traveled: float
    route_completion: float
    max_jerk: float
    max_lateral_acc: float
    mean_speed: float
    traffic_violations: int


@dataclass
class EvaluationMetrics:
    """
    Aggregated evaluation metrics across episodes

    Computes:
    - Safety metrics (collision rate, TTC violations)
    - Progress metrics (route completion, goal reached rate)
    - Comfort metrics (jerk, lateral acceleration)
    - Efficiency metrics (latency, throughput)
    """

    # Raw episode results
    episodes: List[EpisodeResult] = field(default_factory=list)

    def add_episode(self, result: EpisodeResult):
        """Add episode result"""
        self.episodes.append(result)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    # === Safety Metrics ===

    @property
    def collision_rate(self) -> float:
        """Fraction of episodes with collision"""
        if not self.episodes:
            return 0.0
        return sum(e.collision for e in self.episodes) / len(self.episodes)

    @property
    def collision_count(self) -> int:
        """Total number of collisions"""
        return sum(e.collision for e in self.episodes)

    # === Progress Metrics ===

    @property
    def mean_route_completion(self) -> float:
        """Average route completion percentage"""
        if not self.episodes:
            return 0.0
        return np.mean([e.route_completion for e in self.episodes])

    @property
    def goal_reached_rate(self) -> float:
        """Fraction of episodes that reached goal"""
        if not self.episodes:
            return 0.0
        return sum(e.goal_reached for e in self.episodes) / len(self.episodes)

    @property
    def mean_distance_traveled(self) -> float:
        """Average distance traveled"""
        if not self.episodes:
            return 0.0
        return np.mean([e.distance_traveled for e in self.episodes])

    # === Comfort Metrics ===

    @property
    def mean_max_jerk(self) -> float:
        """Average of max jerk across episodes"""
        if not self.episodes:
            return 0.0
        return np.mean([e.max_jerk for e in self.episodes])

    @property
    def jerk_violation_rate(self, threshold: float = 2.0) -> float:
        """Fraction of episodes exceeding jerk threshold"""
        if not self.episodes:
            return 0.0
        return sum(e.max_jerk > threshold for e in self.episodes) / len(self.episodes)

    @property
    def mean_max_lateral_acc(self) -> float:
        """Average of max lateral acceleration across episodes"""
        if not self.episodes:
            return 0.0
        return np.mean([e.max_lateral_acc for e in self.episodes])

    # === Efficiency Metrics ===

    @property
    def mean_reward(self) -> float:
        """Average total reward"""
        if not self.episodes:
            return 0.0
        return np.mean([e.total_reward for e in self.episodes])

    @property
    def mean_episode_length(self) -> float:
        """Average episode length in steps"""
        if not self.episodes:
            return 0.0
        return np.mean([e.steps for e in self.episodes])

    # === Aggregate Methods ===

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        return {
            # Safety
            'collision_rate': self.collision_rate,
            'collision_count': self.collision_count,

            # Progress
            'mean_route_completion': self.mean_route_completion,
            'goal_reached_rate': self.goal_reached_rate,
            'mean_distance_traveled': self.mean_distance_traveled,

            # Comfort
            'mean_max_jerk': self.mean_max_jerk,
            'mean_max_lateral_acc': self.mean_max_lateral_acc,

            # Efficiency
            'mean_reward': self.mean_reward,
            'mean_episode_length': self.mean_episode_length,

            # Meta
            'num_episodes': self.num_episodes
        }

    def check_success_criteria(
        self,
        collision_threshold: float = 0.05,
        completion_threshold: float = 0.85,
        jerk_threshold: float = 2.0
    ) -> Dict[str, bool]:
        """
        Check if metrics meet success criteria

        Default criteria:
        - Collision rate < 5%
        - Route completion > 85%
        - Max jerk < 2 m/s³
        """
        return {
            'collision_rate': self.collision_rate < collision_threshold,
            'route_completion': self.mean_route_completion > completion_threshold,
            'comfort': self.mean_max_jerk < jerk_threshold,
            'overall': (
                self.collision_rate < collision_threshold and
                self.mean_route_completion > completion_threshold and
                self.mean_max_jerk < jerk_threshold
            )
        }

    def __str__(self) -> str:
        """String representation"""
        summary = self.get_summary()
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Episodes: {summary['num_episodes']}",
            "",
            "Safety:",
            f"  Collision Rate: {summary['collision_rate']*100:.1f}%",
            "",
            "Progress:",
            f"  Route Completion: {summary['mean_route_completion']*100:.1f}%",
            f"  Goal Reached Rate: {summary['goal_reached_rate']*100:.1f}%",
            "",
            "Comfort:",
            f"  Mean Max Jerk: {summary['mean_max_jerk']:.2f} m/s³",
            f"  Mean Max Lat Acc: {summary['mean_max_lateral_acc']:.2f} m/s²",
            "",
            "Efficiency:",
            f"  Mean Reward: {summary['mean_reward']:.2f}",
            f"  Mean Episode Length: {summary['mean_episode_length']:.0f} steps",
            "=" * 50
        ]
        return "\n".join(lines)
