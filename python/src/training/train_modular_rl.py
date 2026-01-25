"""
Modular PPO Training Script

Trains ModularDrivingPolicy with support for:
- Frozen encoder modules (preserve learned weights)
- Two-phase training (new encoder → fine-tune all)
- Checkpoint loading with partial weight transfer

Usage:
    # Phase C-1: Add lane encoder to Phase B weights
    python -m src.training.train_modular_rl \
        --config python/configs/planning/modular_ppo_phaseC1.yaml \
        --run-id modular_phaseC1

    # Test with dummy environment
    python -m src.training.train_modular_rl --env dummy --total-steps 10000
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.modular_encoder import (
    ModularEncoderConfig,
    EncoderModuleConfig,
    create_phase_b_config,
    create_lane_encoder_config,
)
from src.models.modular_policy import (
    ModularDrivingPolicy,
    ModularPolicyConfig,
    create_modular_policy_config_phase_b,
)


@dataclass
class ModularPPOConfig:
    """Configuration for Modular PPO training"""

    # Core PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training
    total_steps: int = 2_000_000
    num_steps: int = 2048
    num_epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    lr_schedule: str = "linear"

    # Environment
    num_envs: int = 1
    time_horizon: int = 128

    # Logging
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 20

    # Output
    output_dir: str = "experiments/modular_rl"
    device: str = "cuda"
    run_id: str = "modular_v1"

    # Reward shaping
    reward_scale: float = 1.0
    reward_clip: float = 10.0

    # Two-phase training
    phase_1_steps: Optional[int] = None  # New encoder only
    phase_2_steps: Optional[int] = None  # Fine-tune all
    phase_1_lr: Optional[float] = None
    phase_2_lr: Optional[float] = None

    # Checkpoint loading
    checkpoint_path: Optional[str] = None
    load_encoders: bool = True
    load_fusion: bool = True
    load_heads: bool = False

    # Frozen encoders
    frozen_encoders: List[str] = field(default_factory=list)

    # New encoders to add
    new_encoders: Dict = field(default_factory=dict)


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
        """Yield random mini-batches"""
        total = self.num_steps * self.num_envs
        indices = np.arange(total)
        np.random.shuffle(indices)

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
    """Dummy driving environment for testing"""

    def __init__(self, num_envs: int = 1, obs_dim: int = 254):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = 2

        self.positions = np.zeros((num_envs, 2))
        self.velocities = np.zeros((num_envs, 2))
        self.headings = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=int)
        self.max_steps = 200
        self.goals = np.random.randn(num_envs, 2) * 50
        self.goals[:, 0] = np.abs(self.goals[:, 0]) + 20

    def reset(self) -> np.ndarray:
        self.positions = np.zeros((self.num_envs, 2))
        self.velocities = np.random.randn(self.num_envs, 2) * 0.1
        self.headings = np.zeros(self.num_envs)
        self.steps = np.zeros(self.num_envs, dtype=int)
        self.goals = np.random.randn(self.num_envs, 2) * 50
        self.goals[:, 0] = np.abs(self.goals[:, 0]) + 20
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        steering = actions[:, 0]
        accel = actions[:, 1]

        dt = 0.1
        self.headings += steering * dt
        self.velocities[:, 0] += accel * np.cos(self.headings) * dt
        self.velocities[:, 1] += accel * np.sin(self.headings) * dt

        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(speed > 30, self.velocities / speed * 30, self.velocities)
        self.positions += self.velocities * dt
        self.steps += 1

        dist_to_goal = np.linalg.norm(self.positions - self.goals, axis=1)
        prev_dist = np.linalg.norm(
            (self.positions - self.velocities * dt) - self.goals, axis=1)

        progress_reward = (prev_dist - dist_to_goal) * 1.0
        jerk_penalty = -np.abs(steering) * 0.05
        reached = dist_to_goal < 5.0
        goal_bonus = reached.astype(float) * 10.0

        reward = progress_reward + jerk_penalty + goal_bonus
        done = (self.steps >= self.max_steps) | reached | (dist_to_goal > 200)

        for i in range(self.num_envs):
            if done[i]:
                self.positions[i] = 0
                self.velocities[i] = np.random.randn(2) * 0.1
                self.headings[i] = 0
                self.steps[i] = 0
                self.goals[i] = np.random.randn(2) * 50
                self.goals[i, 0] = np.abs(self.goals[i, 0]) + 20

        return self._get_obs(), reward, done, {'dist_to_goal': dist_to_goal}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        for i in range(self.num_envs):
            obs[i, 0:2] = self.positions[i]
            obs[i, 2:4] = self.velocities[i]
            obs[i, 4] = np.cos(self.headings[i])
            obs[i, 5] = np.sin(self.headings[i])

            dir_to_goal = self.goals[i] - self.positions[i]
            dist = np.linalg.norm(dir_to_goal)
            if dist > 0:
                dir_to_goal /= dist
            obs[i, -30:-28] = dir_to_goal
            obs[i, -28] = min(dist / 100, 1.0)

        return obs


class UnityEnvWrapper:
    """Wrapper for Unity ML-Agents environment"""

    def __init__(self, env_path: Optional[str] = None, port: int = 5004,
                 num_envs: int = 1, obs_dim: int = 254):
        self.obs_dim = obs_dim
        self.action_dim = 2
        self.num_envs = num_envs

        try:
            from mlagents_envs.environment import UnityEnvironment
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )

            engine_channel = EngineConfigurationChannel()
            self.env = UnityEnvironment(
                file_name=env_path,
                base_port=port,
                side_channels=[engine_channel],
                no_graphics=env_path is not None,
            )

            engine_channel.set_configuration_parameters(
                time_scale=20.0,
                target_frame_rate=-1,
                capture_frame_rate=60,
            )

            self.env.reset()

            behavior_names = list(self.env.behavior_specs.keys())
            if not behavior_names:
                raise RuntimeError("No behaviors found in Unity environment")
            self.behavior_name = behavior_names[0]
            self.spec = self.env.behavior_specs[self.behavior_name]

            print(f"Connected to Unity environment")
            print(f"  Behavior: {self.behavior_name}")
            print(f"  Obs shape: {self.spec.observation_specs}")

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

        action_tuple = ActionTuple(continuous=actions)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        obs = self._extract_obs(decision_steps)
        rewards = decision_steps.reward if len(decision_steps) > 0 else np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)

        if len(terminal_steps) > 0:
            for idx in terminal_steps.agent_id:
                env_idx = idx % self.num_envs
                dones[env_idx] = True
                rewards[env_idx] = terminal_steps[idx].reward

        return obs, rewards, dones, {}

    def _extract_obs(self, decision_steps) -> np.ndarray:
        if len(decision_steps) == 0:
            return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)

        obs_list = []
        for obs_spec in decision_steps.obs:
            obs_list.append(obs_spec.reshape(len(decision_steps), -1))

        obs = np.concatenate(obs_list, axis=-1)

        if obs.shape[-1] < self.obs_dim:
            pad = np.zeros((obs.shape[0], self.obs_dim - obs.shape[-1]), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=-1)
        elif obs.shape[-1] > self.obs_dim:
            obs = obs[:, :self.obs_dim]

        return obs.astype(np.float32)

    def close(self):
        self.env.close()


class ModularPPOTrainer:
    """
    PPO Trainer for ModularDrivingPolicy

    Supports:
    - Two-phase training (new encoder → fine-tune all)
    - Frozen encoder detection
    - Checkpoint loading with partial weight transfer
    """

    def __init__(self, policy: ModularDrivingPolicy, env, config: ModularPPOConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )

        self.policy = policy.to(self.device)
        self.env = env

        # Create optimizer with only trainable params
        trainable_params = self.policy.get_trainable_params()
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.lr,
            eps=1e-5,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_dim=policy.total_obs_dim,
            action_dim=2,
        )

        # Training state
        self.total_steps = 0
        self.num_updates = 0
        self.current_phase = 1
        self.episode_rewards = []
        self.episode_lengths = []

        # Output
        self.output_dir = Path(config.output_dir) / config.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_reward': [],
            'episode_length': [],
            'lr': [],
            'phase': [],
        }

    def _print_training_info(self):
        """Print training configuration"""
        print("=" * 60)
        print("MODULAR PPO TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total observation dim: {self.policy.total_obs_dim}")
        print(f"Total parameters: {self.policy.num_parameters:,}")
        print(f"Trainable parameters: {self.policy.num_trainable_parameters:,}")
        print()

        # Encoder status
        print("Encoder Status:")
        for name, status in self.policy.get_encoder_status().items():
            frozen_str = "FROZEN" if status['frozen'] else "TRAINABLE"
            print(f"  {name}: {status['input_dim']}D → {status['output_dim']}D "
                  f"({status['trainable_params']:,} params) [{frozen_str}]")
        print()

        # Phase info
        if self.config.phase_1_steps and self.config.phase_2_steps:
            print("Two-Phase Training:")
            print(f"  Phase 1: {self.config.phase_1_steps:,} steps "
                  f"(lr={self.config.phase_1_lr or self.config.lr:.2e})")
            print(f"  Phase 2: {self.config.phase_2_steps:,} steps "
                  f"(lr={self.config.phase_2_lr or self.config.lr:.2e})")
        else:
            print(f"Single-Phase Training: {self.config.total_steps:,} steps")
        print("=" * 60)

    def train(self) -> Dict:
        """Run full training loop"""
        self._print_training_info()

        obs = self.env.reset()
        start_time = time.time()
        ep_rewards = np.zeros(self.config.num_envs)
        ep_lengths = np.zeros(self.config.num_envs, dtype=int)

        # Determine phase boundaries
        phase_1_steps = self.config.phase_1_steps or self.config.total_steps
        phase_2_steps = self.config.phase_2_steps or 0
        total_steps = phase_1_steps + phase_2_steps

        while self.total_steps < total_steps:
            # Check for phase transition
            if self.current_phase == 1 and self.total_steps >= phase_1_steps and phase_2_steps > 0:
                self._transition_to_phase_2()

            # Collect rollout
            self.buffer.reset()
            for step in range(self.config.num_steps):
                obs_tensor = torch.from_numpy(obs).float().to(self.device)

                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action_and_value(obs_tensor)

                action_np = action.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy().flatten()
                value_np = value.cpu().numpy().flatten()

                next_obs, reward, done, info = self.env.step(action_np)
                reward = np.clip(
                    reward * self.config.reward_scale,
                    -self.config.reward_clip,
                    self.config.reward_clip
                )

                self.buffer.add(obs, action_np, reward, done.astype(float), value_np, log_prob_np)

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
                _, _, last_value = self.policy.get_action_and_value(obs_tensor)
                last_value_np = last_value.cpu().numpy().flatten()

            self.buffer.compute_gae(last_value_np, self.config.gamma, self.config.gae_lambda)

            # PPO update
            update_metrics = self._ppo_update()
            self.num_updates += 1

            # Learning rate schedule
            self._update_lr(total_steps)

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

        # Export ONNX
        onnx_path = self.output_dir / 'policy.onnx'
        self.policy.export_onnx(str(onnx_path))

        return self.history

    def _transition_to_phase_2(self):
        """Transition from Phase 1 to Phase 2 (unfreeze all encoders)"""
        print("\n" + "=" * 60)
        print("TRANSITIONING TO PHASE 2: Unfreezing all encoders")
        print("=" * 60)

        # Save Phase 1 checkpoint
        self._save_checkpoint('phase1_final.pt')

        # Unfreeze all encoders
        self.policy.unfreeze_all_encoders()

        # Recreate optimizer with new learning rate
        trainable_params = self.policy.get_trainable_params()
        new_lr = self.config.phase_2_lr or self.config.lr
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=new_lr,
            eps=1e-5,
        )

        print(f"New trainable parameters: {self.policy.num_trainable_parameters:,}")
        print(f"New learning rate: {new_lr:.2e}")

        self.current_phase = 2

    def _ppo_update(self) -> Dict[str, float]:
        """Perform PPO update"""
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

                new_log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)
                new_log_probs = new_log_probs.squeeze(-1)
                entropy = entropy.squeeze(-1)
                values = values.squeeze(-1)

                # Policy loss
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

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
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

    def _update_lr(self, total_steps: int):
        """Update learning rate based on schedule"""
        if self.config.lr_schedule == "linear":
            progress = self.total_steps / total_steps
            base_lr = self.config.phase_2_lr if self.current_phase == 2 else self.config.lr
            lr = base_lr * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, 1e-7)
        elif self.config.lr_schedule == "cosine":
            import math
            progress = self.total_steps / total_steps
            base_lr = self.config.phase_2_lr if self.current_phase == 2 else self.config.lr
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(lr, 1e-7)

    def _log_update(self, metrics: Dict, start_time: float):
        """Log training progress"""
        elapsed = time.time() - start_time
        fps = self.total_steps / max(elapsed, 1)

        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]

        lr = self.optimizer.param_groups[0]['lr']

        phase_str = f"P{self.current_phase}" if self.config.phase_1_steps else ""

        print(f"Update {self.num_updates:4d} {phase_str}| "
              f"Steps: {self.total_steps:>8,} | "
              f"FPS: {fps:>6.0f} | "
              f"Reward: {np.mean(recent_rewards):>7.2f} | "
              f"EpLen: {np.mean(recent_lengths):>5.0f} | "
              f"PLoss: {metrics['policy_loss']:.4f} | "
              f"VLoss: {metrics['value_loss']:.4f} | "
              f"Ent: {metrics['entropy']:.4f} | "
              f"LR: {lr:.2e}")

        self.history['policy_loss'].append(metrics['policy_loss'])
        self.history['value_loss'].append(metrics['value_loss'])
        self.history['entropy'].append(metrics['entropy'])
        self.history['episode_reward'].append(np.mean(recent_rewards))
        self.history['episode_length'].append(np.mean(recent_lengths))
        self.history['lr'].append(lr)
        self.history['phase'].append(self.current_phase)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        filepath = self.output_dir / filename
        torch.save({
            'total_steps': self.total_steps,
            'num_updates': self.num_updates,
            'current_phase': self.current_phase,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'encoder_status': self.policy.get_encoder_status(),
            'episode_rewards': self.episode_rewards[-1000:],
        }, filepath)

        # Save best model if this is the best reward
        if self.episode_rewards:
            recent_avg = np.mean(self.episode_rewards[-100:])
            best_path = self.output_dir / 'best.pt'
            if not best_path.exists():
                torch.save({
                    'total_steps': self.total_steps,
                    'model_state_dict': self.policy.state_dict(),
                    'avg_reward': recent_avg,
                }, best_path)
            else:
                best_ckpt = torch.load(best_path, weights_only=False)
                if recent_avg > best_ckpt.get('avg_reward', -float('inf')):
                    torch.save({
                        'total_steps': self.total_steps,
                        'model_state_dict': self.policy.state_dict(),
                        'avg_reward': recent_avg,
                    }, best_path)
                    print(f"  ★ New best model: {recent_avg:.2f}")

    def _save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_policy_from_config(config: ModularPPOConfig) -> ModularDrivingPolicy:
    """Create ModularDrivingPolicy from training config"""

    # Start with Phase B config
    encoder_config = create_phase_b_config()

    # Add new encoders if specified
    for name, enc_dict in config.new_encoders.items():
        enc_config = EncoderModuleConfig(
            name=name,
            input_dim=enc_dict['input_dim'],
            hidden_dims=enc_dict.get('hidden_dims', [32]),
            output_dim=enc_dict.get('output_dim', 32),
            frozen=enc_dict.get('frozen', False),
        )
        encoder_config.encoders[name] = enc_config
        encoder_config.encoder_order.append(name)

    # Create policy config
    policy_config = ModularPolicyConfig(
        encoder_config=encoder_config,
        action_dim=2,
    )

    # Create policy
    policy = ModularDrivingPolicy(policy_config)

    # Load checkpoint if specified
    if config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        if config.load_encoders:
            loaded = policy.load_encoder_weights(config.checkpoint_path)
            print(f"  Loaded encoders: {loaded}")
        if config.load_fusion:
            loaded = policy.load_fusion_weights(config.checkpoint_path, match_input_dim=True)
            print(f"  Loaded fusion: {loaded}")
        if config.load_heads:
            loaded = policy.load_head_weights(config.checkpoint_path)
            print(f"  Loaded heads: {loaded}")

    # Freeze specified encoders
    for name in config.frozen_encoders:
        policy.freeze_encoder(name)
        print(f"  Frozen encoder: {name}")

    return policy


def load_config_from_yaml(yaml_path: str) -> ModularPPOConfig:
    """Load ModularPPOConfig from YAML file"""
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Convert nested config to flat ModularPPOConfig
    config_dict = {}

    # Training section
    if 'training' in yaml_config:
        training = yaml_config['training']
        config_dict.update({
            'total_steps': training.get('total_steps', 2_000_000),
            'num_steps': training.get('num_steps', 2048),
            'num_epochs': training.get('num_epochs', 10),
            'batch_size': training.get('batch_size', 256),
            'lr': training.get('lr', 3e-4),
            'lr_schedule': training.get('lr_schedule', 'linear'),
            'gamma': training.get('gamma', 0.99),
            'gae_lambda': training.get('gae_lambda', 0.95),
            'clip_epsilon': training.get('clip_epsilon', 0.2),
        })

        # Two-phase training
        if 'phase_1' in training:
            config_dict['phase_1_steps'] = training['phase_1'].get('steps')
            config_dict['phase_1_lr'] = training['phase_1'].get('lr')
        if 'phase_2' in training:
            config_dict['phase_2_steps'] = training['phase_2'].get('steps')
            config_dict['phase_2_lr'] = training['phase_2'].get('lr')

    # Modular encoder section
    if 'modular_encoder' in yaml_config:
        me = yaml_config['modular_encoder']
        config_dict['frozen_encoders'] = me.get('frozen_encoders', [])
        config_dict['new_encoders'] = me.get('new_encoders', {})

        if 'checkpoint' in me:
            config_dict['checkpoint_path'] = me['checkpoint'].get('path')
            config_dict['load_encoders'] = me['checkpoint'].get('load_encoders', True)
            config_dict['load_fusion'] = me['checkpoint'].get('load_fusion', True)
            config_dict['load_heads'] = me['checkpoint'].get('load_heads', False)

    # Output section
    if 'output' in yaml_config:
        config_dict['output_dir'] = yaml_config['output'].get('dir', 'experiments/modular_rl')
        config_dict['run_id'] = yaml_config['output'].get('run_id', 'modular_v1')

    return ModularPPOConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description='Train ModularDrivingPolicy with PPO')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--env', type=str, default='dummy',
                        choices=['dummy', 'unity'])
    parser.add_argument('--total-steps', type=int, default=2_000_000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='experiments/modular_rl')
    parser.add_argument('--run-id', type=str, default='modular_v1')
    parser.add_argument('--unity-port', type=int, default=5004)
    parser.add_argument('--obs-dim', type=int, default=254,
                        help='Total observation dimension')
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = ModularPPOConfig(
            total_steps=args.total_steps,
            lr=args.lr,
            device=args.device if torch.cuda.is_available() else 'cpu',
            output_dir=args.output_dir,
            run_id=args.run_id,
        )

    # Create environment
    if args.env == 'dummy':
        print("Using dummy environment")
        env = DummyEnv(num_envs=1, obs_dim=args.obs_dim)
    elif args.env == 'unity':
        print("Connecting to Unity ML-Agents environment...")
        env = UnityEnvWrapper(port=args.unity_port, obs_dim=args.obs_dim)

    # Create policy
    policy = create_policy_from_config(config)

    # Create trainer
    trainer = ModularPPOTrainer(policy, env, config)

    # Train
    history = trainer.train()

    if hasattr(env, 'close'):
        env.close()


if __name__ == '__main__':
    main()
