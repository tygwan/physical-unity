"""
Reinforcement Learning (PPO) Training Script

Custom PPO trainer for E2E Driving Model.
Connects to Unity ML-Agents environment for training.

Usage:
    # With Unity Editor (start Unity play mode first)
    python -m src.training.train_rl --env unity

    # With dummy environment (for testing pipeline)
    python -m src.training.train_rl --env dummy --total-steps 10000

    # With pre-trained BC model (CIMRL approach)
    python -m src.training.train_rl --env unity --pretrained experiments/bc/best.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.e2e_model import E2EDrivingModelRL, E2EModelConfig


@dataclass
class PPOConfig:
    """PPO training configuration"""
    # Core PPO hyperparameters
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    clip_epsilon: float = 0.2        # PPO clipping range
    value_loss_coef: float = 0.5     # Value loss weight
    entropy_coef: float = 0.01       # Entropy bonus weight
    max_grad_norm: float = 0.5       # Gradient clipping

    # Training
    total_steps: int = 2_000_000
    num_steps: int = 2048            # Steps per rollout
    num_epochs: int = 10             # Epochs per update
    batch_size: int = 256            # Mini-batch size
    lr: float = 3e-4
    lr_schedule: str = "linear"      # linear, cosine, constant

    # Environment
    num_envs: int = 1                # Parallel environments
    time_horizon: int = 128          # Max steps per episode

    # Logging
    log_interval: int = 10           # Log every N updates
    save_interval: int = 50          # Save every N updates
    eval_interval: int = 20          # Eval every N updates
    eval_episodes: int = 5

    # Output
    output_dir: str = "experiments/rl"
    device: str = "cuda"

    # Reward shaping
    reward_scale: float = 1.0
    reward_clip: float = 10.0


class RolloutBuffer:
    """Stores transitions collected during rollout"""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        self.num_steps = num_steps
        self.num_envs = num_envs

        self.observations = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

        self.step = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.step += 1

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation"""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield random mini-batches from the buffer"""
        total = self.num_steps * self.num_envs
        indices = np.arange(total)
        np.random.shuffle(indices)

        # Flatten
        obs_flat = self.observations.reshape(total, -1)
        actions_flat = self.actions.reshape(total, -1)
        log_probs_flat = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat = self.returns.reshape(total)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        for start in range(0, total, batch_size):
            end = start + batch_size
            if end > total:
                break
            batch_idx = indices[start:end]

            yield {
                'observations': torch.from_numpy(obs_flat[batch_idx]),
                'actions': torch.from_numpy(actions_flat[batch_idx]),
                'old_log_probs': torch.from_numpy(log_probs_flat[batch_idx]),
                'advantages': torch.from_numpy(advantages_flat[batch_idx]),
                'returns': torch.from_numpy(returns_flat[batch_idx]),
            }

    def reset(self):
        self.step = 0


class DummyEnv:
    """
    Dummy driving environment for testing the RL pipeline.
    Simulates a simple 1D driving scenario.
    """

    def __init__(self, num_envs: int = 1, obs_dim: int = 238):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = 2

        # State: position, velocity, heading
        self.positions = np.zeros((num_envs, 2))
        self.velocities = np.zeros((num_envs, 2))
        self.headings = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=int)
        self.max_steps = 200

        # Goal
        self.goals = np.random.randn(num_envs, 2) * 50
        self.goals[:, 0] = np.abs(self.goals[:, 0]) + 20  # Forward

    def reset(self) -> np.ndarray:
        self.positions = np.zeros((self.num_envs, 2))
        self.velocities = np.random.randn(self.num_envs, 2) * 0.1
        self.headings = np.zeros(self.num_envs)
        self.steps = np.zeros(self.num_envs, dtype=int)
        self.goals = np.random.randn(self.num_envs, 2) * 50
        self.goals[:, 0] = np.abs(self.goals[:, 0]) + 20
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Args:
            actions: [num_envs, 2] (steering, acceleration)

        Returns:
            obs, reward, done, info
        """
        steering = actions[:, 0]   # [-0.5, 0.5]
        accel = actions[:, 1]      # [-4.0, 2.0]

        # Update heading
        dt = 0.1
        self.headings += steering * dt

        # Update velocity
        speed = np.linalg.norm(self.velocities, axis=1)
        self.velocities[:, 0] += accel * np.cos(self.headings) * dt
        self.velocities[:, 1] += accel * np.sin(self.headings) * dt

        # Clip speed
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(speed > 30, self.velocities / speed * 30, self.velocities)

        # Update position
        self.positions += self.velocities * dt
        self.steps += 1

        # Compute rewards
        dist_to_goal = np.linalg.norm(self.positions - self.goals, axis=1)
        prev_dist = np.linalg.norm(
            (self.positions - self.velocities * dt) - self.goals, axis=1)

        # Reward: progress toward goal
        progress_reward = (prev_dist - dist_to_goal) * 1.0

        # Penalty: jerk (sudden steering/accel changes)
        jerk_penalty = -np.abs(steering) * 0.05

        # Bonus: reaching goal
        reached = dist_to_goal < 5.0
        goal_bonus = reached.astype(float) * 10.0

        reward = progress_reward + jerk_penalty + goal_bonus

        # Done conditions
        done = (self.steps >= self.max_steps) | reached | (dist_to_goal > 200)

        # Auto-reset done environments
        for i in range(self.num_envs):
            if done[i]:
                self.positions[i] = 0
                self.velocities[i] = np.random.randn(2) * 0.1
                self.headings[i] = 0
                self.steps[i] = 0
                self.goals[i] = np.random.randn(2) * 50
                self.goals[i, 0] = np.abs(self.goals[i, 0]) + 20

        obs = self._get_obs()
        info = {'dist_to_goal': dist_to_goal, 'speed': speed.flatten()}

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Generate 238D observation vector"""
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        for i in range(self.num_envs):
            # Ego state (8D): x, y, vx, vy, cos_h, sin_h, ax, ay
            obs[i, 0:2] = self.positions[i]
            obs[i, 2:4] = self.velocities[i]
            obs[i, 4] = np.cos(self.headings[i])
            obs[i, 5] = np.sin(self.headings[i])

            # Route info (last 30D): direction to goal
            dir_to_goal = self.goals[i] - self.positions[i]
            dist = np.linalg.norm(dir_to_goal)
            if dist > 0:
                dir_to_goal /= dist
            obs[i, -30:-28] = dir_to_goal
            obs[i, -28] = min(dist / 100, 1.0)

        return obs


class UnityEnvWrapper:
    """
    Wrapper for Unity ML-Agents environment.
    Communicates via mlagents_envs gRPC protocol.
    """

    def __init__(self, env_path: Optional[str] = None, port: int = 5004,
                 num_envs: int = 1, obs_dim: int = 238):
        self.obs_dim = obs_dim
        self.action_dim = 2
        self.num_envs = num_envs

        try:
            from mlagents_envs.environment import UnityEnvironment
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )

            # Engine config for faster training
            engine_channel = EngineConfigurationChannel()

            self.env = UnityEnvironment(
                file_name=env_path,  # None = connect to Editor
                base_port=port,
                side_channels=[engine_channel],
                no_graphics=env_path is not None,
            )

            # Speed up simulation
            engine_channel.set_configuration_parameters(
                time_scale=20.0,
                target_frame_rate=-1,
                capture_frame_rate=60,
            )

            self.env.reset()

            # Get behavior name
            behavior_names = list(self.env.behavior_specs.keys())
            if not behavior_names:
                raise RuntimeError("No behaviors found in Unity environment")
            self.behavior_name = behavior_names[0]
            self.spec = self.env.behavior_specs[self.behavior_name]

            print(f"Connected to Unity environment")
            print(f"  Behavior: {self.behavior_name}")
            print(f"  Obs shape: {self.spec.observation_specs}")
            print(f"  Action shape: {self.spec.action_spec}")

        except ImportError:
            raise ImportError(
                "mlagents_envs not installed. Install with: pip install mlagents-envs"
            )

    def reset(self) -> np.ndarray:
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return self._extract_obs(decision_steps)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        from mlagents_envs.base_env import ActionTuple

        # Set actions
        action_tuple = ActionTuple(continuous=actions)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        # Get results
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        obs = self._extract_obs(decision_steps)
        rewards = decision_steps.reward if len(decision_steps) > 0 else np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)

        # Handle terminal agents
        if len(terminal_steps) > 0:
            for idx in terminal_steps.agent_id:
                env_idx = idx % self.num_envs
                dones[env_idx] = True
                rewards[env_idx] = terminal_steps[idx].reward

        return obs, rewards, dones, {}

    def _extract_obs(self, decision_steps) -> np.ndarray:
        """Extract observation from Unity decision steps"""
        if len(decision_steps) == 0:
            return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        # Concatenate all observation tensors
        obs_list = []
        for obs_spec in decision_steps.obs:
            obs_list.append(obs_spec.reshape(len(decision_steps), -1))

        obs = np.concatenate(obs_list, axis=-1)

        # Pad or truncate to obs_dim
        if obs.shape[-1] < self.obs_dim:
            pad = np.zeros((obs.shape[0], self.obs_dim - obs.shape[-1]), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=-1)
        elif obs.shape[-1] > self.obs_dim:
            obs = obs[:, :self.obs_dim]

        return obs.astype(np.float32)

    def close(self):
        self.env.close()


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer

    Supports:
    - Custom E2EDrivingModelRL policy
    - Unity ML-Agents environment connection
    - BC pre-training initialization (CIMRL)
    - GAE for advantage estimation
    """

    def __init__(self, model: E2EDrivingModelRL, env, config: PPOConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )

        self.model = model.to(self.device)
        self.env = env

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            eps=1e-5,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_dim=model.config.total_obs_dim,
            action_dim=model.config.action_dim,
        )

        # Training state
        self.total_steps = 0
        self.num_updates = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_reward': [],
            'episode_length': [],
            'lr': [],
        }

    def train(self) -> Dict:
        """Run full PPO training loop"""
        print(f"Starting PPO training on {self.device}")
        print(f"  Model params: {self.model.num_trainable_parameters:,}")
        print(f"  Total steps: {self.config.total_steps:,}")
        print(f"  Steps per rollout: {self.config.num_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Num epochs: {self.config.num_epochs}")
        print("-" * 60)

        obs = self.env.reset()
        start_time = time.time()
        ep_rewards = np.zeros(self.config.num_envs)
        ep_lengths = np.zeros(self.config.num_envs, dtype=int)

        while self.total_steps < self.config.total_steps:
            # Collect rollout
            self.buffer.reset()
            for step in range(self.config.num_steps):
                obs_tensor = torch.from_numpy(obs).float().to(self.device)

                with torch.no_grad():
                    action, log_prob, value = self.model.get_action_and_value(obs_tensor)

                action_np = action.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy().flatten()
                value_np = value.cpu().numpy().flatten()

                # Step environment
                next_obs, reward, done, info = self.env.step(action_np)

                # Clip reward
                reward = np.clip(reward * self.config.reward_scale, -self.config.reward_clip, self.config.reward_clip)

                # Store transition
                self.buffer.add(obs, action_np, reward, done.astype(float), value_np, log_prob_np)

                # Track episodes
                ep_rewards += reward
                ep_lengths += 1
                for i in range(self.config.num_envs):
                    if done[i]:
                        self.episode_rewards.append(ep_rewards[i])
                        self.episode_lengths.append(ep_lengths[i])
                        ep_rewards[i] = 0
                        ep_lengths[i] = 0

                obs = next_obs
                self.total_steps += self.config.num_envs

            # Compute returns and advantages
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                _, _, last_value = self.model.get_action_and_value(obs_tensor)
                last_value_np = last_value.cpu().numpy().flatten()

            self.buffer.compute_gae(last_value_np, self.config.gamma, self.config.gae_lambda)

            # PPO update
            update_metrics = self._ppo_update()
            self.num_updates += 1

            # Learning rate schedule
            self._update_lr()

            # Logging
            if self.num_updates % self.config.log_interval == 0:
                self._log_update(update_metrics, start_time)

            # Save checkpoint
            if self.num_updates % self.config.save_interval == 0:
                self._save_checkpoint(f'step_{self.total_steps}.pt')

        # Final save
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Total steps: {self.total_steps:,}")
        self._save_checkpoint('final.pt')
        self._save_history()

        return self.history

    def _ppo_update(self) -> Dict[str, float]:
        """Perform PPO update on collected rollout"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)

                # Evaluate actions with current policy
                new_log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                new_log_probs = new_log_probs.squeeze(-1)
                entropy = entropy.squeeze(-1)
                values = values.squeeze(-1)

                # Policy loss (PPO clipped objective)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + self.config.value_loss_coef * value_loss
                        + self.config.entropy_coef * entropy_loss)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_batches += 1

        return {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'entropy': total_entropy / max(num_batches, 1),
        }

    def _update_lr(self):
        """Update learning rate based on schedule"""
        if self.config.lr_schedule == "linear":
            progress = self.total_steps / self.config.total_steps
            lr = self.config.lr * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, 1e-7)
        elif self.config.lr_schedule == "cosine":
            import math
            progress = self.total_steps / self.config.total_steps
            lr = self.config.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, 1e-7)

    def _log_update(self, metrics: Dict, start_time: float):
        """Log training progress"""
        elapsed = time.time() - start_time
        fps = self.total_steps / max(elapsed, 1)

        # Recent episode stats
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]

        lr = self.optimizer.param_groups[0]['lr']

        print(f"Update {self.num_updates:4d} | "
              f"Steps: {self.total_steps:>8,} | "
              f"FPS: {fps:>6.0f} | "
              f"Reward: {np.mean(recent_rewards):>7.2f} | "
              f"EpLen: {np.mean(recent_lengths):>5.0f} | "
              f"PLoss: {metrics['policy_loss']:.4f} | "
              f"VLoss: {metrics['value_loss']:.4f} | "
              f"Ent: {metrics['entropy']:.4f} | "
              f"LR: {lr:.2e}")

        # Store history
        self.history['policy_loss'].append(metrics['policy_loss'])
        self.history['value_loss'].append(metrics['value_loss'])
        self.history['entropy'].append(metrics['entropy'])
        self.history['episode_reward'].append(np.mean(recent_rewards))
        self.history['episode_length'].append(np.mean(recent_lengths))
        self.history['lr'].append(lr)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        filepath = self.output_dir / filename
        torch.save({
            'total_steps': self.total_steps,
            'num_updates': self.num_updates,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config,
            'model_config': vars(self.model.config),
            'episode_rewards': self.episode_rewards[-1000:],
        }, filepath)

    def _save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def from_bc_pretrained(cls, checkpoint_path: str, env,
                           config: Optional[PPOConfig] = None) -> 'PPOTrainer':
        """
        Create PPO trainer from BC pre-trained model (CIMRL approach).

        This loads a BC-trained model and continues with RL fine-tuning.
        """
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        model_config = E2EModelConfig(**ckpt['model_config'])
        model = E2EDrivingModelRL(model_config)

        # Load BC weights (skip value head since it's new)
        model_state = model.state_dict()
        bc_state = ckpt['model_state_dict']
        for key in bc_state:
            if key in model_state and model_state[key].shape == bc_state[key].shape:
                model_state[key] = bc_state[key]
        model.load_state_dict(model_state)

        print(f"Loaded BC pre-trained weights from {checkpoint_path}")
        print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")
        print(f"  Best val loss: {ckpt.get('best_val_loss', 'unknown')}")

        config = config or PPOConfig()
        return cls(model, env, config)


def main():
    parser = argparse.ArgumentParser(description='Train E2E Driving Model with PPO')
    parser.add_argument('--env', type=str, default='dummy',
                        choices=['dummy', 'unity'],
                        help='Environment type')
    parser.add_argument('--total-steps', type=int, default=2_000_000)
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='experiments/rl')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to BC pre-trained checkpoint (CIMRL)')
    parser.add_argument('--unity-port', type=int, default=5004)
    args = parser.parse_args()

    # PPO config
    ppo_config = PPOConfig(
        total_steps=args.total_steps,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device if torch.cuda.is_available() else 'cpu',
        output_dir=args.output_dir,
    )

    # Create environment
    if args.env == 'dummy':
        print("Using dummy environment for pipeline testing")
        env = DummyEnv(num_envs=1, obs_dim=238)
    elif args.env == 'unity':
        print("Connecting to Unity ML-Agents environment...")
        env = UnityEnvWrapper(port=args.unity_port, obs_dim=238)
    else:
        raise ValueError(f"Unknown env: {args.env}")

    # Create trainer
    if args.pretrained:
        # CIMRL: Load BC model and fine-tune with RL
        trainer = PPOTrainer.from_bc_pretrained(
            args.pretrained, env, ppo_config
        )
    else:
        # Train from scratch
        model_config = E2EModelConfig(
            level=args.level,
            hidden_dims=[512, 512, 256],
            predict_trajectory=False,
        )
        model = E2EDrivingModelRL(model_config)
        print(f"Model created: {model.num_trainable_parameters:,} parameters")
        trainer = PPOTrainer(model, env, ppo_config)

    # Train
    history = trainer.train()

    # Export best model
    if hasattr(env, 'close'):
        env.close()


if __name__ == '__main__':
    main()
