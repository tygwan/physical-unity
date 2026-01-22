"""
Base Trainer class for all training pipelines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import mlflow


@dataclass
class TrainingConfig:
    """Base training configuration"""
    # Experiment
    experiment_name: str = "ad_planning"
    run_name: str = "default"

    # Training
    total_steps: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 3e-4

    # Evaluation
    eval_interval: int = 10_000
    eval_episodes: int = 10

    # Checkpointing
    checkpoint_interval: int = 50_000
    checkpoint_dir: str = "experiments/checkpoints"

    # Logging
    log_interval: int = 1000
    use_tensorboard: bool = True
    use_mlflow: bool = True
    use_wandb: bool = False

    # Device
    device: str = "cuda"


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers

    Provides common functionality:
    - Experiment tracking (MLflow/W&B)
    - Checkpointing
    - Logging
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        # Setup experiment tracking
        if config.use_mlflow:
            mlflow.set_experiment(config.experiment_name)
            self.run = mlflow.start_run(run_name=config.run_name)
            mlflow.log_params(self._config_to_dict(config))

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to experiment tracker"""
        step = step or self.global_step

        if self.config.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        # Print to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step}: {metrics_str}")

    def _config_to_dict(self, config: TrainingConfig) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_')
        }

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'config') and self.config.use_mlflow:
            mlflow.end_run()


class RLTrainer(BaseTrainer):
    """
    Trainer for RL algorithms (PPO, SAC)

    Works with Unity ML-Agents environment.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env,
        policy,
        algorithm: str = "ppo"
    ):
        super().__init__(config)
        self.env = env
        self.policy = policy.to(self.device)
        self.algorithm = algorithm

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

    def train(self) -> Dict[str, Any]:
        """Main RL training loop"""
        print(f"Starting {self.algorithm.upper()} training...")
        print(f"Device: {self.device}")
        print(f"Total steps: {self.config.total_steps}")

        results = {
            'best_reward': float('-inf'),
            'final_reward': 0.0
        }

        while self.global_step < self.config.total_steps:
            # Collect experience
            # ... (implementation depends on algorithm)

            # Update policy
            # ... (PPO or SAC update)

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                # Log training metrics
                pass

            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.log_metrics(eval_metrics)

                if eval_metrics.get('mean_reward', 0) > results['best_reward']:
                    results['best_reward'] = eval_metrics['mean_reward']
                    self.save_checkpoint(
                        f"{self.config.checkpoint_dir}/best_model.pt"
                    )

            # Checkpointing
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/checkpoint_{self.global_step}.pt"
                )

        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy"""
        self.policy.eval()

        total_reward = 0.0
        total_collisions = 0
        total_completions = 0

        with torch.no_grad():
            for _ in range(self.config.eval_episodes):
                # Run evaluation episode
                # ... (implementation)
                pass

        self.policy.train()

        return {
            'mean_reward': total_reward / self.config.eval_episodes,
            'collision_rate': total_collisions / self.config.eval_episodes,
            'completion_rate': total_completions / self.config.eval_episodes
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'global_step': self.global_step,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self._config_to_dict(self.config)
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint['global_step']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")


class BCTrainer(BaseTrainer):
    """
    Trainer for Behavioral Cloning

    Uses expert demonstrations from datasets.
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset,
        policy
    ):
        super().__init__(config)
        self.dataset = dataset
        self.policy = policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        self.criterion = torch.nn.MSELoss()

    def train(self) -> Dict[str, Any]:
        """Main BC training loop"""
        print("Starting Behavioral Cloning training...")

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        results = {'best_loss': float('inf')}

        for epoch in range(self.config.total_steps):
            epoch_loss = 0.0

            for batch in dataloader:
                obs, actions = batch
                obs = {k: v.to(self.device) for k, v in obs.items()}
                actions = actions.to(self.device)

                # Forward pass
                pred_actions = self.policy(
                    obs['ego_state'],
                    obs['route_info'],
                    obs['surrounding']
                )

                # Compute loss
                loss = self.criterion(pred_actions, actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.log_metrics({'loss': avg_loss}, step=epoch)

            if avg_loss < results['best_loss']:
                results['best_loss'] = avg_loss
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/bc_best.pt"
                )

        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        # Implementation
        return {}

    def save_checkpoint(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
