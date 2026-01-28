"""
Hybrid RL Training Script for Phase C-1

Uses HybridDrivingPolicy to train with frozen Phase B encoder
while learning to use new lane observations.

Two-Phase Training:
  Phase 1 (500K steps): Lane encoder + combiner only
  Phase 2 (1.5M steps): Fine-tune all (unfreeze Phase B)

Usage:
    # Phase C-1 training (starts from Phase B checkpoint)
    python -m src.training.train_hybrid_rl \
        --checkpoint results/phase-B/E2EDrivingAgent/E2EDrivingAgent-2000150.pt \
        --run-id hybrid_phaseC1

    # Resume from hybrid checkpoint
    python -m src.training.train_hybrid_rl \
        --resume experiments/hybrid_rl/hybrid_phaseC1/checkpoint.pt

    # Fine-tune mode (Phase 2)
    python -m src.training.train_hybrid_rl \
        --resume experiments/hybrid_rl/hybrid_phaseC1/checkpoint.pt \
        --unfreeze-phase-b
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'python'))

from src.models.hybrid_policy import (
    HybridDrivingPolicy,
    HybridPolicyConfig,
    create_hybrid_policy_config_phase_c1
)


class PPOBuffer:
    """Rollout buffer for PPO training."""

    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, action, reward, value, log_prob, done):
        """Store a single transition."""
        assert self.ptr < self.buffer_size
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0):
        """Compute GAE advantages for the path."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)

        # GAE-Lambda
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - self.dones[path_slice]) - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)

        # Returns
        self.returns[path_slice] = self.advantages[path_slice] + values[:-1]

        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum."""
        result = np.zeros_like(x)
        running_sum = 0
        for t in reversed(range(len(x))):
            running_sum = x[t] + discount * running_sum
            result[t] = running_sum
        return result

    def get(self):
        """Get all data and reset buffer."""
        assert self.ptr == self.buffer_size

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

        data = dict(
            obs=torch.FloatTensor(self.obs),
            actions=torch.FloatTensor(self.actions),
            returns=torch.FloatTensor(self.returns),
            advantages=torch.FloatTensor(self.advantages),
            log_probs=torch.FloatTensor(self.log_probs)
        )

        self.ptr = 0
        self.path_start_idx = 0

        return data


class HybridPPOTrainer:
    """
    PPO Trainer for Hybrid Policy.

    Supports two-phase training:
    - Phase 1: Frozen Phase B, train lane encoder + combiner
    - Phase 2: Unfreeze all for fine-tuning
    """

    def __init__(
        self,
        policy: HybridDrivingPolicy,
        env=None,  # Unity environment
        lr: float = 1.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_steps: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = None
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
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Optimizer (only trainable params)
        self.optimizer = optim.Adam(
            policy.get_trainable_params(),
            lr=lr,
            eps=1e-5
        )

        # Logging
        self.log_dir = log_dir
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Training state
        self.global_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Buffer
        self.buffer = PPOBuffer(
            obs_dim=policy.total_obs_dim,
            action_dim=policy.config.action_dim,
            buffer_size=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

    def collect_rollout_dummy(self, num_steps: int):
        """
        Dummy rollout collection for testing without Unity.
        Generates random observations and rewards.
        """
        print(f"Collecting {num_steps} steps (dummy mode)...")

        obs = np.random.randn(self.policy.total_obs_dim).astype(np.float32)

        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy().flatten()
            log_prob_np = log_prob.cpu().numpy().flatten()[0]
            value_np = value.cpu().numpy().flatten()[0]

            # Dummy environment step
            next_obs = np.random.randn(self.policy.total_obs_dim).astype(np.float32)
            reward = np.random.randn() * 0.1  # Small random reward
            done = np.random.random() < 0.01  # 1% episode end

            self.buffer.store(obs, action_np, reward, value_np, log_prob_np, done)

            if done:
                self.buffer.finish_path(0.0)
                obs = np.random.randn(self.policy.total_obs_dim).astype(np.float32)
                self.episode_rewards.append(reward)
                self.episode_lengths.append(step)
            else:
                obs = next_obs

        # Finish last path
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action_and_value(obs_tensor)
            self.buffer.finish_path(last_value.cpu().numpy().flatten()[0])

    def update(self) -> Dict[str, float]:
        """Perform PPO update step."""
        data = self.buffer.get()

        obs = data['obs'].to(self.device)
        actions = data['actions'].to(self.device)
        returns = data['returns'].to(self.device)
        advantages = data['advantages'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)

        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        num_samples = len(obs)
        indices = np.arange(num_samples)

        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Policy loss
                ratio = torch.exp(new_log_probs.squeeze() - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

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
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def train(
        self,
        total_steps: int,
        phase_1_steps: int = 500000,
        log_interval: int = 10,
        save_interval: int = 50,
        save_dir: str = None
    ):
        """
        Main training loop.

        Args:
            total_steps: Total training steps
            phase_1_steps: Steps for Phase 1 (frozen Phase B)
            log_interval: Steps between logging
            save_interval: Steps between checkpoint saves
            save_dir: Directory to save checkpoints
        """
        print(f"Starting training for {total_steps} steps")
        print(f"  Phase 1 (frozen): {phase_1_steps} steps")
        print(f"  Phase 2 (fine-tune): {total_steps - phase_1_steps} steps")
        print(f"  Trainable params: {self.policy.num_trainable_params:,}")

        num_iterations = total_steps // self.num_steps
        phase_1_iterations = phase_1_steps // self.num_steps
        start_time = time.time()

        for iteration in range(num_iterations):
            # Phase transition check
            if iteration == phase_1_iterations:
                print(f"\n=== Transitioning to Phase 2 (unfreezing Phase B) ===")
                self.policy.unfreeze_phase_b()

                # Reset optimizer with new params and lower LR
                self.optimizer = optim.Adam(
                    self.policy.get_trainable_params(),
                    lr=self.lr * 0.2,  # Lower LR for fine-tuning
                    eps=1e-5
                )
                print(f"  New trainable params: {self.policy.num_trainable_params:,}")

            # Collect rollout
            self.collect_rollout_dummy(self.num_steps)

            # Update
            metrics = self.update()
            self.global_step += self.num_steps

            # Logging
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed

                phase = "Phase 1" if iteration < phase_1_iterations else "Phase 2"
                print(f"[{phase}] Step {self.global_step:,} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Value Loss: {metrics['value_loss']:.4f} | "
                      f"Entropy: {metrics['entropy']:.4f} | "
                      f"SPS: {sps:.0f}")

                if self.writer:
                    self.writer.add_scalar('train/policy_loss', metrics['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', metrics['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', metrics['entropy'], self.global_step)

            # Save checkpoint
            if save_dir and iteration % save_interval == 0 and iteration > 0:
                self.save_checkpoint(save_dir, f'checkpoint_{self.global_step}.pt')

        print(f"\nTraining complete! Total steps: {self.global_step:,}")

    def save_checkpoint(self, save_dir: str, filename: str):
        """Save training checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, filename)

        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.policy.config.__dict__,
            'phase_b_frozen': self.policy.phase_b_encoder.is_frozen
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

        if checkpoint.get('phase_b_frozen', True):
            self.policy.freeze_phase_b()
        else:
            self.policy.unfreeze_phase_b()

        print(f"Loaded checkpoint from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid PPO Training for Phase C-1')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Phase B checkpoint path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from hybrid checkpoint')
    parser.add_argument('--run-id', type=str, default='hybrid_phaseC1',
                       help='Run identifier')
    parser.add_argument('--total-steps', type=int, default=2000000,
                       help='Total training steps')
    parser.add_argument('--phase-1-steps', type=int, default=500000,
                       help='Phase 1 steps (frozen Phase B)')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                       help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--unfreeze-phase-b', action='store_true',
                       help='Start with Phase B unfrozen (Phase 2 mode)')
    parser.add_argument('--dummy', action='store_true',
                       help='Use dummy environment (for testing)')

    args = parser.parse_args()

    # Setup directories
    run_dir = f'experiments/hybrid_rl/{args.run_id}'
    log_dir = f'{run_dir}/logs'
    save_dir = f'{run_dir}/checkpoints'
    os.makedirs(run_dir, exist_ok=True)

    # Create policy
    config = create_hybrid_policy_config_phase_c1()

    if args.resume:
        print(f"Resuming from: {args.resume}")
        policy = HybridDrivingPolicy(config)
        trainer = HybridPPOTrainer(
            policy=policy,
            lr=args.lr,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            log_dir=log_dir
        )
        trainer.load_checkpoint(args.resume)

    elif args.checkpoint:
        print(f"Loading Phase B checkpoint: {args.checkpoint}")
        policy = HybridDrivingPolicy.from_phase_b_checkpoint(
            checkpoint_path=args.checkpoint,
            config=config,
            freeze_phase_b=not args.unfreeze_phase_b
        )
        trainer = HybridPPOTrainer(
            policy=policy,
            lr=args.lr,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            log_dir=log_dir
        )

    else:
        print("Warning: No checkpoint provided, starting from scratch")
        policy = HybridDrivingPolicy(config)
        trainer = HybridPPOTrainer(
            policy=policy,
            lr=args.lr,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            log_dir=log_dir
        )

    # Train (dummy mode for now)
    if args.dummy:
        print("\n=== Running in DUMMY mode (no Unity) ===\n")
        trainer.train(
            total_steps=min(args.total_steps, 50000),  # Shorter for testing
            phase_1_steps=min(args.phase_1_steps, 25000),
            log_interval=5,
            save_interval=10,
            save_dir=save_dir
        )
    else:
        print("\n=== Unity environment required for real training ===")
        print("Use --dummy flag for testing without Unity")
        print("Or connect Unity with ML-Agents to start training")


if __name__ == '__main__':
    main()
