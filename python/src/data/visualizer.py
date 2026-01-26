"""
Scenario Visualizer
====================

Visualizes driving scenarios for data inspection and debugging.

Features:
- Bird's eye view plot of trajectories
- Ego + agents animation
- Speed/acceleration profiles
- Scenario statistics

Usage:
    from src.data.visualizer import ScenarioVisualizer

    viz = ScenarioVisualizer()
    viz.plot_scenario(scenario)
    viz.plot_ego_profile(scenario)
    viz.plot_batch_stats(scenarios)
"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

from .base import Scenario

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization disabled.")


class ScenarioVisualizer:
    """Visualizes driving scenarios."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), save_dir: Optional[str] = None):
        self.figsize = figsize
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_scenario(
        self,
        scenario: Scenario,
        show_agents: bool = True,
        show_ego_history: bool = True,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
    ):
        """
        Plot bird's eye view of a scenario.

        Args:
            scenario: Scenario to visualize
            show_agents: Show other agent trajectories
            show_ego_history: Show ego past trajectory
            title: Plot title
            save_name: File name to save (without extension)
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for visualization")
            return

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        ego = scenario.ego_trajectory

        # Plot ego trajectory
        ax.plot(ego[:, 0], ego[:, 1], 'b-', linewidth=2, label='Ego', zorder=5)
        ax.plot(ego[0, 0], ego[0, 1], 'go', markersize=10, label='Start', zorder=6)
        ax.plot(ego[-1, 0], ego[-1, 1], 'r*', markersize=12, label='End', zorder=6)

        # Plot ego vehicle rectangle at start
        self._draw_vehicle(ax, ego[0], color='blue', alpha=0.5)

        # Plot agents
        if show_agents and scenario.agents:
            colors = plt.cm.Set2(np.linspace(0, 1, min(len(scenario.agents), 8)))
            for i, agent in enumerate(scenario.agents[:20]):  # Limit to 20 agents
                color = colors[i % len(colors)]
                traj = agent.trajectory

                # Filter valid positions (non-zero)
                valid = np.any(traj[:, :2] != 0, axis=1)
                if not np.any(valid):
                    continue

                valid_traj = traj[valid]
                ax.plot(
                    valid_traj[:, 0], valid_traj[:, 1],
                    '-', color=color, linewidth=1, alpha=0.7
                )

                # Draw vehicle at first valid position
                self._draw_vehicle(ax, valid_traj[0], color=color, alpha=0.3, size=0.7)

        # Formatting
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(
                f"Scenario: {scenario.scenario_id} | "
                f"Duration: {scenario.duration:.1f}s | "
                f"Agents: {len(scenario.agents)}"
            )

        plt.tight_layout()
        self._save_or_show(fig, save_name)

    def plot_ego_profile(
        self,
        scenario: Scenario,
        save_name: Optional[str] = None,
    ):
        """
        Plot ego vehicle speed, acceleration, and steering profiles.
        """
        if not HAS_MATPLOTLIB:
            return

        ego = scenario.ego_trajectory
        dt = 0.1  # 10 Hz
        time = np.arange(ego.shape[0]) * dt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Speed
        speed = np.sqrt(ego[:, 2]**2 + ego[:, 3]**2)
        axes[0].plot(time, speed * 3.6, 'b-', linewidth=1.5)  # km/h
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f"Ego Profile - {scenario.scenario_id}")

        # Acceleration
        acc = np.sqrt(ego[:, 4]**2 + ego[:, 5]**2)
        # Sign based on velocity direction
        speed_diff = np.diff(speed, prepend=speed[0])
        acc_signed = acc * np.sign(speed_diff)
        axes[1].plot(time, acc_signed, 'r-', linewidth=1.5)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].grid(True, alpha=0.3)

        # Heading
        axes[2].plot(time, np.degrees(ego[:, 6]), 'g-', linewidth=1.5)
        axes[2].set_ylabel('Heading (degrees)')
        axes[2].grid(True, alpha=0.3)

        # Heading rate (steering proxy)
        heading_rate = np.diff(ego[:, 6], prepend=ego[0, 6]) / dt
        # Unwrap large jumps
        heading_rate = np.clip(heading_rate, -2, 2)
        axes[3].plot(time, heading_rate, 'm-', linewidth=1.5)
        axes[3].set_ylabel('Heading Rate (rad/s)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_or_show(fig, save_name)

    def plot_batch_stats(
        self,
        scenarios: List[Scenario],
        save_name: Optional[str] = None,
    ):
        """
        Plot statistics across a batch of scenarios.
        """
        if not HAS_MATPLOTLIB:
            return

        # Compute statistics
        durations = []
        agent_counts = []
        avg_speeds = []
        max_speeds = []

        for s in scenarios:
            durations.append(s.duration)
            agent_counts.append(len(s.agents))

            speed = np.sqrt(s.ego_trajectory[:, 2]**2 + s.ego_trajectory[:, 3]**2)
            avg_speeds.append(np.mean(speed) * 3.6)  # km/h
            max_speeds.append(np.max(speed) * 3.6)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Duration distribution
        axes[0, 0].hist(durations, bins=30, color='steelblue', edgecolor='white')
        axes[0, 0].set_xlabel('Duration (s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Scenario Duration Distribution')

        # Agent count distribution
        axes[0, 1].hist(agent_counts, bins=range(0, max(agent_counts)+2),
                       color='coral', edgecolor='white')
        axes[0, 1].set_xlabel('Number of Agents')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Agent Count Distribution')

        # Speed distribution
        axes[1, 0].hist(avg_speeds, bins=30, color='seagreen', edgecolor='white',
                       alpha=0.7, label='Average')
        axes[1, 0].hist(max_speeds, bins=30, color='darkgreen', edgecolor='white',
                       alpha=0.5, label='Max')
        axes[1, 0].set_xlabel('Speed (km/h)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Speed Distribution')
        axes[1, 0].legend()

        # Summary text
        axes[1, 1].axis('off')
        stats_text = (
            f"Dataset Statistics\n"
            f"{'='*30}\n"
            f"Total scenarios: {len(scenarios)}\n"
            f"Total duration: {sum(durations)/3600:.1f} hours\n"
            f"\n"
            f"Duration:\n"
            f"  Mean: {np.mean(durations):.1f}s\n"
            f"  Min: {np.min(durations):.1f}s\n"
            f"  Max: {np.max(durations):.1f}s\n"
            f"\n"
            f"Agents per scenario:\n"
            f"  Mean: {np.mean(agent_counts):.1f}\n"
            f"  Max: {np.max(agent_counts)}\n"
            f"\n"
            f"Ego speed:\n"
            f"  Avg: {np.mean(avg_speeds):.1f} km/h\n"
            f"  Max: {np.max(max_speeds):.1f} km/h\n"
        )
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.suptitle(f"Batch Statistics ({len(scenarios)} scenarios)", fontsize=14)
        plt.tight_layout()
        self._save_or_show(fig, save_name)

    def _draw_vehicle(
        self,
        ax,
        state: np.ndarray,
        color: str = 'blue',
        alpha: float = 0.5,
        size: float = 1.0,
    ):
        """Draw a vehicle rectangle at given state."""
        x, y = state[0], state[1]
        heading = state[6] if len(state) > 6 else 0

        length = 4.5 * size
        width = 1.8 * size

        # Vehicle corners (centered)
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2],
        ])

        # Rotate
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = corners @ rot.T

        # Translate
        corners[:, 0] += x
        corners[:, 1] += y

        polygon = patches.Polygon(corners, closed=True,
                                 facecolor=color, edgecolor='black',
                                 alpha=alpha, linewidth=0.5)
        ax.add_patch(polygon)

    def _save_or_show(self, fig, save_name: Optional[str]):
        """Save figure or show interactively."""
        if save_name and self.save_dir:
            path = self.save_dir / f"{save_name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot: {path}")
            plt.close(fig)
        else:
            plt.show()


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy scenario
    T = 150
    t = np.linspace(0, 15, T)

    ego = np.zeros((T, 7), dtype=np.float32)
    ego[:, 0] = t * 5 + 0.5 * np.sin(t * 0.5)  # x with slight curve
    ego[:, 1] = 2 * np.sin(t * 0.3)              # y with lane change
    ego[:, 2] = 5 + 0.5 * np.cos(t * 0.5)        # vx
    ego[:, 3] = 0.6 * np.cos(t * 0.3)            # vy
    ego[:, 6] = np.arctan2(ego[:, 3], ego[:, 2])  # heading

    agents = []
    for i in range(5):
        a_traj = np.zeros((T, 7), dtype=np.float32)
        a_traj[:, 0] = ego[:, 0] + np.random.uniform(-10, 20)
        a_traj[:, 1] = ego[:, 1] + np.random.uniform(-7, 7)
        a_traj[:, 2] = np.random.uniform(3, 8)
        agents.append(AgentTrack(f"agent_{i}", "vehicle", a_traj, (4.5, 1.8, 1.5)))

    from .base import AgentTrack
    scenario = Scenario("viz_test", "test", 15.0, ego, agents)

    viz = ScenarioVisualizer(save_dir="outputs/viz")
    viz.plot_scenario(scenario, save_name="test_scenario")
    viz.plot_ego_profile(scenario, save_name="test_profile")
    viz.plot_batch_stats([scenario] * 10, save_name="test_stats")
