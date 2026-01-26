"""
Imitation Learning (Behavioral Cloning) Training Script

Trains E2E driving model to imitate expert driving from nuPlan data.

Usage:
    python -m src.training.train_il --config configs/planning/bc_config.yaml
    python -m src.training.train_il --epochs 100 --batch-size 256 --lr 3e-4
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.e2e_model import E2EDrivingModel, E2EModelConfig


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for pre-processed samples"""

    def __init__(self, samples):
        """
        Args:
            samples: dict with 'observations' and 'actions' numpy arrays
                     Optionally includes 'images' for Level 2+
        """
        if isinstance(samples, dict):
            self.observations = torch.as_tensor(samples['observations'], dtype=torch.float32)
            self.actions = torch.as_tensor(samples['actions'], dtype=torch.float32)
            self.images = None
            if 'images' in samples:
                self.images = torch.as_tensor(samples['images'], dtype=torch.float32)
        elif isinstance(samples, list):
            import numpy as np
            obs = np.array([s['observation'] for s in samples], dtype=np.float32)
            act = np.array([s['action'] for s in samples], dtype=np.float32)
            self.observations = torch.from_numpy(obs)
            self.actions = torch.from_numpy(act)
            self.images = None
        else:
            raise ValueError(f"Unsupported samples type: {type(samples)}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        item = {
            'observation': self.observations[idx],
            'action': self.actions[idx],
        }
        if self.images is not None:
            item['images'] = self.images[idx]
        return item


class BCTrainer:
    """Behavioral Cloning Trainer for E2E Driving Model"""

    def __init__(
        self,
        model: E2EDrivingModel,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        config: Optional[Dict] = None,
    ):
        self.config = config or self._default_config()
        self.device = torch.device(
            self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True,
            )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config.get('lr_min', 1e-6),
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'experiments/bc'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_config() -> Dict:
        return {
            'batch_size': 256,
            'epochs': 100,
            'lr': 3e-4,
            'lr_min': 1e-6,
            'weight_decay': 1e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'early_stopping_patience': 15,
            'save_every': 10,
            'log_every': 10,
            'output_dir': 'experiments/bc',
            'gradient_clip': 1.0,
            # Loss weights
            'steering_weight': 0.5,
            'acceleration_weight': 0.5,
            'trajectory_weight': 0.2,
        }

    def train(self) -> Dict:
        """Run full training loop"""
        print(f"Starting BC training on {self.device}")
        print(f"  Model params: {self.model.num_trainable_parameters:,}")
        print(f"  Train samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            print(f"  Val samples: {len(self.val_dataset):,}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  LR: {self.config['lr']}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.epoch = epoch

            # Train
            train_metrics = self._train_epoch()
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Validate
            val_metrics = {}
            if self.val_dataset:
                val_metrics = self._validate()
                self.history['val_loss'].append(val_metrics['total_loss'])

                # Early stopping
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.patience_counter = 0
                    self._save_checkpoint('best.pt')
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # Scheduler step
            self.scheduler.step()

            # Logging
            if (epoch + 1) % self.config['log_every'] == 0 or epoch == 0:
                self._log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt')

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final model
        self._save_checkpoint('final.pt')
        self._save_history()

        return self.history

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'steering_loss': 0.0,
            'acceleration_loss': 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            obs = batch['observation'].to(self.device)
            action = batch['action'].to(self.device)
            images = batch.get('images')
            if images is not None:
                images = images.to(self.device)

            # Forward pass
            losses = self.model.compute_bc_loss(obs, action, images=images)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            if self.config.get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

        # Average
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'steering_loss': 0.0,
            'acceleration_loss': 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            obs = batch['observation'].to(self.device)
            action = batch['action'].to(self.device)
            images = batch.get('images')
            if images is not None:
                images = images.to(self.device)

            losses = self.model.compute_bc_loss(obs, action, images=images)

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics"""
        lr = self.optimizer.param_groups[0]['lr']
        msg = (f"Epoch {epoch + 1:3d}/{self.config['epochs']} | "
               f"Train Loss: {train_metrics['total_loss']:.6f} | "
               f"Steer: {train_metrics['steering_loss']:.6f} | "
               f"Accel: {train_metrics['acceleration_loss']:.6f}")

        if val_metrics:
            msg += (f" | Val Loss: {val_metrics['total_loss']:.6f}"
                    f" | Best: {self.best_val_loss:.6f}")

        msg += f" | LR: {lr:.2e}"
        print(msg)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        filepath = self.output_dir / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': vars(self.model.config),
        }, filepath)

    def _save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_dummy_dataset(num_samples: int = 1000, obs_dim: int = 238,
                         level: int = 1, num_cameras: int = 4,
                         image_size: int = 64) -> SimpleDataset:
    """Create dummy dataset for testing the training pipeline

    Args:
        num_samples: Number of samples to generate
        obs_dim: Observation vector dimension
        level: Model level (1=vector only, 2+=with images)
        num_cameras: Number of cameras (Level 2+)
        image_size: Image height/width (Level 2+)
    """
    import numpy as np

    observations = np.random.randn(num_samples, obs_dim).astype(np.float32)
    actions = np.random.randn(num_samples, 2).astype(np.float32)
    # Scale actions to realistic range
    actions[:, 0] = np.clip(actions[:, 0] * 0.2, -0.5, 0.5)   # steering
    actions[:, 1] = np.clip(actions[:, 1] * 2.0, -4.0, 2.0)    # acceleration

    data = {
        'observations': observations,
        'actions': actions,
    }

    if level >= 2:
        # Generate random images (small for testing)
        images = np.random.randn(
            num_samples, num_cameras, 3, image_size, image_size
        ).astype(np.float32) * 0.1
        data['images'] = images

    return SimpleDataset(data)


def main():
    parser = argparse.ArgumentParser(description='Train E2E Driving Model with Behavioral Cloning')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='experiments/bc')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2],
                        help='Model level (1=MLP, 2=CNN)')
    parser.add_argument('--num-cameras', type=int, default=4,
                        help='Number of cameras (Level 2)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Camera image size (Level 2)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to processed nuPlan data')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy data for testing pipeline')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export best model to ONNX after training')
    args = parser.parse_args()

    # Model config
    model_config = E2EModelConfig(
        level=args.level,
        hidden_dims=[512, 512, 256],
        dropout=0.1,
        predict_trajectory=False,
        num_cameras=args.num_cameras,
        image_size=(args.image_size, args.image_size),
    )

    # Create model
    model = E2EDrivingModel(model_config)
    print(f"Model created: {model.num_trainable_parameters:,} parameters")

    # Load data
    if args.dummy:
        print(f"Using dummy dataset for pipeline testing (Level {args.level})")
        img_size = 64 if args.level >= 2 else 0  # Small images for testing
        train_dataset = create_dummy_dataset(
            5000 if args.level == 1 else 200,
            level=args.level,
            num_cameras=args.num_cameras,
            image_size=img_size,
        )
        val_dataset = create_dummy_dataset(
            1000 if args.level == 1 else 50,
            level=args.level,
            num_cameras=args.num_cameras,
            image_size=img_size,
        )
    elif args.data_dir:
        # Load processed nuPlan data
        data_path = Path(args.data_dir)
        train_data = torch.load(data_path / 'train_samples.pt', weights_only=False)
        val_data = torch.load(data_path / 'val_samples.pt', weights_only=False)
        train_dataset = SimpleDataset(train_data)
        val_dataset = SimpleDataset(val_data)
    else:
        print("No data specified. Use --dummy for testing or --data-dir for real data.")
        return

    # Training config
    train_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        'output_dir': args.output_dir,
        'early_stopping_patience': 15,
        'save_every': 10,
        'log_every': 5,
        'gradient_clip': 1.0,
    }

    # Train
    trainer = BCTrainer(model, train_dataset, val_dataset, train_config)
    history = trainer.train()

    # Export to ONNX
    if args.export_onnx:
        onnx_path = Path(args.output_dir) / 'model.onnx'
        # Load best model
        best_ckpt = torch.load(Path(args.output_dir) / 'best.pt', weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        model.export_onnx(str(onnx_path))
        print(f"ONNX model exported to {onnx_path}")


if __name__ == '__main__':
    main()
