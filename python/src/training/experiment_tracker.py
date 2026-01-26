"""
Experiment Tracking for Autonomous Driving ML Platform.
Supports MLflow and Weights & Biases for logging metrics, artifacts, and hyperparameters.
"""

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

# Optional imports for experiment trackers
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    # Experiment info
    name: str = "vehicle_training"
    project: str = "ad-platform"
    tags: list = field(default_factory=list)
    notes: str = ""

    # Tracker selection
    use_mlflow: bool = True
    use_wandb: bool = False

    # MLflow settings
    mlflow_tracking_uri: str = "file:./experiments/mlruns"
    mlflow_experiment_name: str = "VehicleAgent"

    # W&B settings
    wandb_project: str = "ad-platform"
    wandb_entity: Optional[str] = None

    # Logging settings
    log_dir: str = "./experiments/logs"
    checkpoint_dir: str = "./experiments/checkpoints"
    artifact_dir: str = "./experiments/artifacts"


class ExperimentTracker:
    """
    Unified experiment tracking interface for MLflow and W&B.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_id = None
        self.run_name = None
        self._mlflow_run = None
        self._wandb_run = None

        # Create directories
        for dir_path in [config.log_dir, config.checkpoint_dir, config.artifact_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def start_run(self, run_name: Optional[str] = None, hyperparameters: Optional[Dict] = None):
        """Start a new experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"{self.config.name}_{timestamp}"

        # Start MLflow run
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            self.run_id = self._mlflow_run.info.run_id

            if hyperparameters:
                mlflow.log_params(self._flatten_dict(hyperparameters))

            if self.config.tags:
                mlflow.set_tags({"tags": ",".join(self.config.tags)})

            print(f"[ExperimentTracker] MLflow run started: {self.run_name} (ID: {self.run_id})")

        # Start W&B run
        if self.config.use_wandb and WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.run_name,
                config=hyperparameters,
                tags=self.config.tags,
                notes=self.config.notes
            )
            print(f"[ExperimentTracker] W&B run started: {self._wandb_run.url}")

        return self.run_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all enabled trackers."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.log_metrics(metrics, step=step)

        if self.config.use_wandb and WANDB_AVAILABLE and self._wandb_run:
            wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters/hyperparameters."""
        flat_params = self._flatten_dict(params)

        if self.config.use_mlflow and MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.log_params(flat_params)

        if self.config.use_wandb and WANDB_AVAILABLE and self._wandb_run:
            wandb.config.update(params)

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log an artifact (model, data, etc.)."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.log_artifact(artifact_path)

        if self.config.use_wandb and WANDB_AVAILABLE and self._wandb_run:
            artifact = wandb.Artifact(
                name=Path(artifact_path).stem,
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

    def log_model(self, model_path: str, model_name: str = "model"):
        """Log a trained model."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.log_artifact(model_path, artifact_path="models")

        if self.config.use_wandb and WANDB_AVAILABLE and self._wandb_run:
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def end_run(self, status: str = "FINISHED"):
        """End the current experiment run."""
        if self.config.use_mlflow and MLFLOW_AVAILABLE and self._mlflow_run:
            mlflow.end_run(status=status)
            print(f"[ExperimentTracker] MLflow run ended: {status}")

        if self.config.use_wandb and WANDB_AVAILABLE and self._wandb_run:
            wandb.finish()
            print(f"[ExperimentTracker] W&B run ended")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_tracker_from_config(config_path: str = None) -> ExperimentTracker:
    """Create experiment tracker from configuration file."""
    # Default config
    exp_config = ExperimentConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
            for key, value in loaded.items():
                if hasattr(exp_config, key):
                    setattr(exp_config, key, value)

    return ExperimentTracker(exp_config)


# Example usage
if __name__ == "__main__":
    # Create tracker
    config = ExperimentConfig(
        name="vehicle_ppo_test",
        use_mlflow=True,
        use_wandb=False
    )
    tracker = ExperimentTracker(config)

    # Start run with hyperparameters
    hyperparams = {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "batch_size": 2048,
        "network": {
            "hidden_units": 512,
            "num_layers": 3
        }
    }
    tracker.start_run(hyperparameters=hyperparams)

    # Log some metrics
    for step in range(0, 1000, 100):
        tracker.log_metrics({
            "reward": step * 0.1,
            "loss": 1.0 / (step + 1)
        }, step=step)

    # End run
    tracker.end_run()

    print("Experiment tracking test completed!")
