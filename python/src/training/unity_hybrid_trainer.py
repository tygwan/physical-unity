"""
Unity Hybrid Trainer for Phase C-1

Connects to Unity ML-Agents environment and trains HybridDrivingPolicy
with frozen Phase B encoder and trainable lane encoder.

Usage:
    # Start Unity in Play mode first, then:
    python -m python.src.training.unity_hybrid_trainer \
        --checkpoint results/phase-B/E2EDrivingAgent/E2EDrivingAgent-2000150.pt \
        --run-id hybrid_phaseC1
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ML-Agents environment
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

from src.models.hybrid_policy import (
    HybridDrivingPolicy,
    HybridPolicyConfig,
    create_hybrid_policy_config_phase_c1
)


class UnityVecEnv:
    """
    Vectorized Unity environment wrapper for PPO training.
    Handles multiple agents across training areas.
    """

    def __init__(self, env: UnityEnvironment, behavior_name: str = None):
        self.env = env
        self.env.reset()

        # Get behavior name
        if behavior_name is None:
            behavior_names = list(self.env.behavior_specs.keys())
            if len(behavior_names) == 0:
                raise ValueError("No behaviors found in Unity environment")
            behavior_name = behavior_names[0]

        self.behavior_name = behavior_name
        self.spec = self.env.behavior_specs[behavior_name]

        # Get observation and action dimensions
        self.obs_dim = sum(spec.shape[0] for spec in self.spec.observation_specs)
        self.action_dim = self.spec.action_spec.continuous_size

        print(f"Unity Environment Connected:")
        print(f"  Behavior: {self.behavior_name}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")

        # Track agent states
        self.agent_ids = []
        self.num_agents = 0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observations."""
        self.env.reset()
        return self._get_observations()

    def _get_observations(self) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
        """
        Get observations from Unity environment.

        Returns:
            obs: (num_agents, obs_dim) array
            agent_ids: List of agent IDs
            rewards: (num_agents,) array
            dones: (num_agents,) array
        """
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # Combine observations from all agents
        all_obs = []
        all_ids = []
        all_rewards = []
        all_dones = []

        # Decision steps (agents requesting actions)
        for agent_id in decision_steps.agent_id:
            idx = decision_steps.agent_id_to_index[agent_id]
            obs = np.concatenate([
                decision_steps.obs[i][idx] for i in range(len(decision_steps.obs))
            ])
            all_obs.append(obs)
            all_ids.append(agent_id)
            all_rewards.append(decision_steps.reward[idx])
            all_dones.append(False)

        # Terminal steps (episode ended)
        for agent_id in terminal_steps.agent_id:
            idx = terminal_steps.agent_id_to_index[agent_id]
            obs = np.concatenate([
                terminal_steps.obs[i][idx] for i in range(len(terminal_steps.obs))
            ])
            all_obs.append(obs)
            all_ids.append(agent_id)
            all_rewards.append(terminal_steps.reward[idx])
            all_dones.append(True)

        if len(all_obs) == 0:
            return np.zeros((0, self.obs_dim)), [], np.zeros(0), np.zeros(0, dtype=bool)

        return (
            np.array(all_obs, dtype=np.float32),
            all_ids,
            np.array(all_rewards, dtype=np.float32),
            np.array(all_dones, dtype=bool)
        )

    def step(self, actions: np.ndarray, agent_ids: List[int]) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
        """
        Send actions to Unity and get next observations.

        Args:
            actions: (num_agents, action_dim) array
            agent_ids: List of agent IDs corresponding to actions

        Returns:
            next_obs, next_agent_ids, rewards, dones
        """
        # Get decision steps to check how many agents are waiting
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        num_agents_waiting = len(decision_steps)

        if num_agents_waiting > 0 and len(actions) > 0:
            # Match actions to waiting agents
            # Unity expects actions for agents in decision_steps, in order
            num_actions_to_send = min(len(actions), num_agents_waiting)
            actions_to_send = actions[:num_actions_to_send]

            # Create action tuple for Unity
            action_tuple = ActionTuple(continuous=actions_to_send)

            # Set actions for agents
            self.env.set_actions(self.behavior_name, action_tuple)

        # Step environment
        self.env.step()

        # Get next observations
        return self._get_observations()

    def close(self):
        """Close Unity environment."""
        self.env.close()


class UnityHybridTrainer:
    """
    PPO Trainer using Unity environment with HybridDrivingPolicy.
    """

    def __init__(
        self,
        policy: HybridDrivingPolicy,
        env: UnityVecEnv,
        lr: float = 1.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        batch_size: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.policy = policy.to(device)
        self.env = env
        self.device = device

        # PPO hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(
            policy.get_trainable_params(),
            lr=lr,
            eps=1e-5
        )

        # Training state
        self.global_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Per-agent episode tracking
        self.agent_episode_rewards = {}
        self.agent_episode_lengths = {}

    def collect_rollout(self, num_steps: int) -> Dict:
        """
        Collect rollout data from Unity environment.

        Returns:
            Dictionary with obs, actions, rewards, values, log_probs, dones, advantages, returns
        """
        # Storage
        all_obs = []
        all_actions = []
        all_rewards = []
        all_values = []
        all_log_probs = []
        all_dones = []

        # Per-agent pending transitions (obs, action, value, log_prob)
        pending_transitions = {}

        steps_collected = 0
        empty_steps = 0
        max_empty_steps = 1000  # Prevent infinite loop

        while steps_collected < num_steps and empty_steps < max_empty_steps:
            # Get current state from Unity
            decision_steps, terminal_steps = self.env.env.get_steps(self.env.behavior_name)

            # Process terminal agents first (episodes ended)
            if len(terminal_steps) > 0 and steps_collected < 10:
                print(f"[DEBUG] Terminal steps: {len(terminal_steps)} agents ended episodes")
            for agent_id in terminal_steps.agent_id:
                idx = terminal_steps.agent_id_to_index[agent_id]
                reward = terminal_steps.reward[idx]

                if agent_id in pending_transitions:
                    # Complete the pending transition
                    trans = pending_transitions.pop(agent_id)
                    all_obs.append(trans['obs'])
                    all_actions.append(trans['action'])
                    all_rewards.append(reward)
                    all_values.append(trans['value'])
                    all_log_probs.append(trans['log_prob'])
                    all_dones.append(True)

                    # Track episode stats
                    self.agent_episode_rewards[agent_id] = self.agent_episode_rewards.get(agent_id, 0.0) + reward
                    self.agent_episode_lengths[agent_id] = self.agent_episode_lengths.get(agent_id, 0) + 1

                    self.episode_rewards.append(self.agent_episode_rewards[agent_id])
                    self.episode_lengths.append(self.agent_episode_lengths[agent_id])

                    # Reset for new episode
                    self.agent_episode_rewards[agent_id] = 0.0
                    self.agent_episode_lengths[agent_id] = 0

                    steps_collected += 1

            # Process decision-requesting agents
            num_decision_agents = len(decision_steps)

            if num_decision_agents == 0:
                # No agents waiting, step environment and try again
                self.env.env.step()
                empty_steps += 1
                continue

            empty_steps = 0  # Reset empty counter

            # Get observations for decision agents
            obs_list = []
            agent_id_list = list(decision_steps.agent_id)

            for agent_id in agent_id_list:
                idx = decision_steps.agent_id_to_index[agent_id]
                obs = np.concatenate([
                    decision_steps.obs[i][idx] for i in range(len(decision_steps.obs))
                ])
                obs_list.append(obs)

                # If agent has pending transition, complete it with intermediate reward
                reward = decision_steps.reward[idx]
                if agent_id in pending_transitions:
                    trans = pending_transitions.pop(agent_id)
                    all_obs.append(trans['obs'])
                    all_actions.append(trans['action'])
                    all_rewards.append(reward)
                    all_values.append(trans['value'])
                    all_log_probs.append(trans['log_prob'])
                    all_dones.append(False)

                    # Track episode stats
                    self.agent_episode_rewards[agent_id] = self.agent_episode_rewards.get(agent_id, 0.0) + reward
                    self.agent_episode_lengths[agent_id] = self.agent_episode_lengths.get(agent_id, 0) + 1

                    steps_collected += 1

            if len(obs_list) == 0:
                self.env.env.step()
                continue

            obs_array = np.array(obs_list, dtype=np.float32)

            # Debug: print first observation periodically
            if steps_collected < 5:
                print(f"\n[DEBUG] Obs sample (first 20 of {obs_array.shape[1]}): {obs_array[0, :20]}")
                print(f"[DEBUG] Obs stats: min={obs_array.min():.3f}, max={obs_array.max():.3f}, mean={obs_array.mean():.3f}")

            # Get actions from policy
            obs_tensor = torch.FloatTensor(obs_array).to(self.device)
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action_and_value(obs_tensor)

            # Debug: print actions periodically
            if steps_collected < 5:
                actions_clamped_debug = np.clip(actions.cpu().numpy(), -1.0, 1.0)
                print(f"[DEBUG] Actions (clamped): {actions_clamped_debug[0]}")

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy().flatten()
            values_np = values.cpu().numpy().flatten()

            # Store pending transitions with UNCLAMPED actions for training consistency
            # (log_prob was computed on unclamped actions, must match during evaluate)
            for i, agent_id in enumerate(agent_id_list):
                pending_transitions[agent_id] = {
                    'obs': obs_list[i],
                    'action': actions_np[i],  # Unclamped for training
                    'value': values_np[i],
                    'log_prob': log_probs_np[i]
                }

            # Clamp actions to [-1, 1] before sending to Unity
            actions_clamped = np.clip(actions_np, -1.0, 1.0)
            action_tuple = ActionTuple(continuous=actions_clamped)
            self.env.env.set_actions(self.env.behavior_name, action_tuple)
            self.env.env.step()

        # Handle case where no complete transitions were collected
        if len(all_obs) == 0:
            print(f"Warning: No complete transitions collected. Pending: {len(pending_transitions)}")
            # Return minimal rollout to avoid crashes
            obs_dim = self.env.obs_dim
            action_dim = self.env.action_dim
            return {
                'obs': torch.zeros((1, obs_dim)),
                'actions': torch.zeros((1, action_dim)),
                'rewards': torch.zeros(1),
                'values': torch.zeros(1),
                'log_probs': torch.zeros(1),
                'dones': torch.zeros(1, dtype=torch.bool),
                'advantages': torch.zeros(1),
                'returns': torch.zeros(1)
            }

        # Convert to arrays
        all_obs = np.array(all_obs, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)
        all_rewards = np.array(all_rewards, dtype=np.float32)
        all_values = np.array(all_values, dtype=np.float32)
        all_log_probs = np.array(all_log_probs, dtype=np.float32)
        all_dones = np.array(all_dones, dtype=bool)

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(all_rewards, all_values, all_dones)

        return {
            'obs': torch.FloatTensor(all_obs),
            'actions': torch.FloatTensor(all_actions),
            'rewards': torch.FloatTensor(all_rewards),
            'values': torch.FloatTensor(all_values),
            'log_probs': torch.FloatTensor(all_log_probs),
            'dones': torch.BoolTensor(all_dones),
            'advantages': torch.FloatTensor(advantages),
            'returns': torch.FloatTensor(returns)
        }

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, rollout: Dict) -> Dict[str, float]:
        """Perform PPO update."""
        obs = rollout['obs'].to(self.device)
        actions = rollout['actions'].to(self.device)
        old_log_probs = rollout['log_probs'].to(self.device)
        advantages = rollout['advantages'].to(self.device)
        returns = rollout['returns'].to(self.device)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        num_samples = len(obs)
        indices = np.arange(num_samples)

        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs.squeeze() - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.get_trainable_params(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1)
        }

    def train(
        self,
        total_steps: int,
        stage_steps: List[int] = None,
        rollout_steps: int = 2048,
        log_interval: int = 10,
        save_interval: int = 100,
        save_dir: str = None,
        value_warmup_threshold: float = 100.0,
        gradual_unfreeze: bool = True
    ):
        """
        6-Stage Gradual Unfreezing Training Loop:
        - Stage 0 (value warmup): Only value_head trainable, VLoss < threshold
        - Stage 1 (lane encoder): + lane_encoder
        - Stage 2 (combiner gate): + combiner.gate
        - Stage 3 (full combiner): + combiner.lane_proj
        - Stage 4 (policy head): + policy_head + log_std
        - Stage 5 (fine-tune): + Phase B encoder (optional, lower LR)

        Args:
            total_steps: Total training steps
            stage_steps: Steps for each stage [s0, s1, s2, s3, s4, s5]
                         Default: [50k, 200k, 200k, 200k, 500k, remaining]
            rollout_steps: Steps per rollout
            log_interval: Logging interval
            save_interval: Checkpoint save interval
            save_dir: Directory to save checkpoints
            value_warmup_threshold: VLoss threshold for Stage 0 completion
            gradual_unfreeze: Use gradual unfreezing (True) or original 3-phase (False)
        """
        # Default stage steps for gradual unfreezing
        if stage_steps is None:
            if gradual_unfreeze:
                # Gradual: 50k + 200k + 200k + 200k + 500k + remaining
                stage_steps = [50000, 200000, 200000, 200000, 500000, total_steps]
            else:
                # Original 3-phase: 50k (value) + 500k (adapt) + remaining (fine-tune)
                stage_steps = [50000, 0, 0, 0, 500000, total_steps]

        stage_names = [
            "Stage 0 (value warmup)",
            "Stage 1 (+ lane_encoder)",
            "Stage 2 (+ combiner.gate)",
            "Stage 3 (+ combiner.lane_proj)",
            "Stage 4 (+ policy_head)",
            "Stage 5 (+ Phase B encoder)"
        ]

        print(f"\n{'='*70}")
        print(f"Starting Hybrid PPO Training (6-Stage Gradual Unfreezing)")
        print(f"{'='*70}")
        print(f"Total steps: {total_steps:,}")
        for i, (name, steps) in enumerate(zip(stage_names, stage_steps)):
            if steps > 0 or i == 0:
                print(f"  {name}: {steps:,} steps")
        print(f"Trainable params (initial): {self.policy.num_trainable_params:,}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        num_iterations = total_steps // rollout_steps
        start_time = time.time()

        # Calculate cumulative stage iterations
        cumulative_steps = [0]
        for steps in stage_steps:
            cumulative_steps.append(cumulative_steps[-1] + steps)

        current_stage = 0
        recent_value_losses = deque(maxlen=10)
        recent_rewards = deque(maxlen=50)

        for iteration in range(num_iterations):
            current_step = iteration * rollout_steps

            # Check for stage transitions
            if current_stage < 5:
                should_advance = False

                # Stage 0: Check value loss threshold
                if current_stage == 0:
                    avg_vloss = np.mean(recent_value_losses) if recent_value_losses else float('inf')
                    if (avg_vloss < value_warmup_threshold and len(recent_value_losses) >= 5) or \
                       current_step >= cumulative_steps[1]:
                        should_advance = True

                # Other stages: Check step count
                elif current_step >= cumulative_steps[current_stage + 1]:
                    should_advance = True

                if should_advance:
                    current_stage += 1
                    self._apply_stage_unfreezing(current_stage, stage_names)

            # Collect rollout
            rollout = self.collect_rollout(rollout_steps)
            self.global_step += rollout_steps

            # Update policy
            metrics = self.update(rollout)

            # Track metrics for stage transitions
            recent_value_losses.append(metrics['value_loss'])
            if self.episode_rewards:
                recent_rewards.append(np.mean(list(self.episode_rewards)[-10:]))

            # Logging
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / max(elapsed, 1)

                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

                print(f"[Stage{current_stage}] Step {self.global_step:>8,} | "
                      f"Reward: {mean_reward:>7.1f} | "
                      f"Length: {mean_length:>5.0f} | "
                      f"PLoss: {metrics['policy_loss']:.4f} | "
                      f"VLoss: {metrics['value_loss']:.4f} | "
                      f"Ent: {metrics['entropy']:.3f} | "
                      f"SPS: {sps:.0f}")

            # Save checkpoint
            if save_dir and iteration % save_interval == 0 and iteration > 0:
                self.save_checkpoint(save_dir, f'checkpoint_{self.global_step}.pt')

        # Final save
        if save_dir:
            self.save_checkpoint(save_dir, 'checkpoint_final.pt')
            self.export_onnx(save_dir)

        print(f"\nTraining complete! Total steps: {self.global_step:,}")

    def _apply_stage_unfreezing(self, stage: int, stage_names: List[str]):
        """Apply unfreezing for the given stage."""
        print(f"\n{'='*70}")
        print(f"Advancing to {stage_names[stage]}")
        print(f"{'='*70}")

        lr_multiplier = 1.0

        if stage == 1:
            # Stage 1: Unfreeze lane_encoder
            self.policy.unfreeze_lane_encoder()
        elif stage == 2:
            # Stage 2: Unfreeze combiner gate
            self.policy.unfreeze_combiner_gate()
        elif stage == 3:
            # Stage 3: Unfreeze combiner lane_proj
            self.policy.unfreeze_combiner_lane_proj()
        elif stage == 4:
            # Stage 4: Unfreeze policy_head
            self.policy.unfreeze_policy_head()
        elif stage == 5:
            # Stage 5: Unfreeze Phase B encoder with lower LR
            self.policy.unfreeze_phase_b()
            lr_multiplier = 0.1  # Much lower LR for fine-tuning

        # Recreate optimizer with updated trainable params
        current_lr = self.lr * lr_multiplier
        self.optimizer = optim.Adam(
            self.policy.get_trainable_params(),
            lr=current_lr,
            eps=1e-5
        )

        print(f"Trainable params: {self.policy.num_trainable_params:,}")
        print(f"Learning rate: {current_lr:.2e}\n")

    def save_checkpoint(self, save_dir: str, filename: str):
        """Save checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.policy.config.__dict__,
            'phase_b_frozen': self.policy.phase_b_encoder.is_frozen,
            'episode_rewards': list(self.episode_rewards),
        }, path)

        print(f"  Saved: {path}")

    def export_onnx(self, save_dir: str):
        """Export policy to ONNX."""
        path = os.path.join(save_dir, 'hybrid_policy.onnx')
        self.policy.export_onnx(path)


def main():
    parser = argparse.ArgumentParser(description='Unity Hybrid PPO Training (Gradual Unfreezing)')
    parser.add_argument('--checkpoint', type=str,
                       default='results/phase-B/E2EDrivingAgent/E2EDrivingAgent-2000150.pt',
                       help='Phase B checkpoint path')
    parser.add_argument('--run-id', type=str, default='hybrid_phaseC1_gradual',
                       help='Run identifier')
    parser.add_argument('--total-steps', type=int, default=3000000,
                       help='Total training steps (longer for gradual unfreezing)')
    parser.add_argument('--stage-steps', type=str, default='50000,200000,200000,200000,500000',
                       help='Comma-separated steps for stages 0-4 (stage 5 uses remaining)')
    parser.add_argument('--rollout-steps', type=int, default=4096,
                       help='Steps per rollout')
    parser.add_argument('--lr', type=float, default=1.0e-4,
                       help='Learning rate (slightly lower for stability)')
    parser.add_argument('--port', type=int, default=5004,
                       help='Unity environment port')
    parser.add_argument('--time-scale', type=float, default=20.0,
                       help='Unity time scale')
    parser.add_argument('--env', type=str, default=None,
                       help='Path to Unity build executable (None for Editor)')
    parser.add_argument('--no-graphics', action='store_true',
                       help='Run without graphics (headless)')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--bypass-combiner', action='store_true',
                       help='Bypass combiner (pure Phase B behavior for debugging)')
    parser.add_argument('--no-gradual', action='store_true',
                       help='Disable gradual unfreezing (use original 3-phase)')

    args = parser.parse_args()

    # Setup directories
    run_dir = f'experiments/hybrid_rl/{args.run_id}'
    save_dir = f'{run_dir}/checkpoints'
    os.makedirs(run_dir, exist_ok=True)

    # Connect to Unity
    env_path = args.env
    if env_path:
        print(f"Launching Unity build: {env_path}")
    else:
        print(f"Connecting to Unity Editor on port {args.port}...")

    try:
        # Build additional args for the environment
        additional_args = ['--mlagents-port', str(args.port)]
        if args.time_scale != 1.0:
            additional_args.extend(['--time-scale', str(args.time_scale)])

        # Create environment parameters side channel
        env_params_channel = EnvironmentParametersChannel()

        unity_env = UnityEnvironment(
            file_name=env_path,  # None for Editor, path for build
            base_port=args.port,
            no_graphics=args.no_graphics,
            timeout_wait=300,
            additional_args=additional_args if env_path else None,
            num_areas=args.num_envs,
            side_channels=[env_params_channel]
        )

        # Set curriculum parameters to match Phase B training (VerySlow curriculum start)
        print("\nSetting curriculum parameters (Phase B Fast config - matching training end):")
        env_params_channel.set_float_parameter("num_active_npcs", 1.0)
        env_params_channel.set_float_parameter("npc_speed_ratio", 0.9)  # Fast NPC - Phase B trained to lesson 3
        env_params_channel.set_float_parameter("goal_distance", 150.0)
        env_params_channel.set_float_parameter("speed_zone_count", 1.0)
        env_params_channel.set_float_parameter("npc_speed_variation", 0.1)
        env_params_channel.set_float_parameter("lane_marking_difficulty", 1.0)
        print("  num_active_npcs: 1.0")
        print("  npc_speed_ratio: 0.9 (Fast - Phase B final stage)")
        print("  goal_distance: 150.0")
        print("  speed_zone_count: 1.0")
    except Exception as e:
        print(f"Failed to connect to Unity: {e}")
        if env_path:
            print(f"\nMake sure the build exists at: {env_path}")
        else:
            print("\nMake sure Unity is running in Play mode!")
        return

    # Wrap environment
    env = UnityVecEnv(unity_env)

    # Verify observation dimension
    if env.obs_dim != 254:
        print(f"Warning: Expected 254D observations, got {env.obs_dim}D")

    # Create policy
    print(f"\nLoading Phase B checkpoint: {args.checkpoint}")
    config = create_hybrid_policy_config_phase_c1()
    policy = HybridDrivingPolicy.from_phase_b_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        freeze_phase_b=True,
        freeze_policy_head=True,  # Critical: freeze policy_head during value warmup
        freeze_combiner=True  # Critical: freeze combiner to preserve Phase B behavior
    )

    # Enable bypass mode for debugging if requested
    if args.bypass_combiner:
        policy.set_bypass_combiner(True)

    # Create trainer
    trainer = UnityHybridTrainer(
        policy=policy,
        env=env,
        lr=args.lr,
    )

    # Parse stage steps
    stage_steps = [int(s) for s in args.stage_steps.split(',')]
    # Add remaining steps for stage 5
    stage_steps.append(args.total_steps)

    # Train
    try:
        trainer.train(
            total_steps=args.total_steps,
            stage_steps=stage_steps,
            rollout_steps=args.rollout_steps,
            log_interval=10,
            save_interval=100,
            save_dir=save_dir,
            gradual_unfreeze=not args.no_gradual
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(save_dir, 'checkpoint_interrupted.pt')
    finally:
        env.close()
        print("Unity environment closed.")


if __name__ == '__main__':
    main()
