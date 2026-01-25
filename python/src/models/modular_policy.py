"""
Modular Driving Policy for Autonomous Driving

Integrates ModularEncoder with Actor-Critic architecture for PPO/SAC training.
Supports incremental learning through modular encoder expansion.

Example usage:
    # Create policy with Phase B encoder
    config = create_phase_b_config()
    policy = ModularDrivingPolicy(config)

    # Load Phase B weights
    policy.load_encoder_weights("results/phaseB/best.pt")
    policy.load_fusion_weights("results/phaseB/best.pt")

    # Add lane encoder for Phase C-1
    lane_config = create_lane_encoder_config()
    policy.add_encoder(lane_config, freeze_existing=True)

    # Train with frozen encoders, only lane encoder + heads trainable
    optimizer = Adam(policy.get_trainable_params(), lr=1.5e-4)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .modular_encoder import (
    ModularEncoder,
    ModularEncoderConfig,
    EncoderModuleConfig,
    get_activation,
)


@dataclass
class ModularPolicyConfig:
    """Configuration for ModularDrivingPolicy"""

    # Encoder config
    encoder_config: ModularEncoderConfig = field(default_factory=ModularEncoderConfig)

    # Action space
    action_dim: int = 2                 # [steering, acceleration]
    steering_range: Tuple[float, float] = (-0.5, 0.5)    # radians
    accel_range: Tuple[float, float] = (-4.0, 2.0)       # m/s^2

    # Actor-Critic architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.1

    # Stochastic policy
    log_std_init: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Trajectory prediction (auxiliary task)
    predict_trajectory: bool = False
    trajectory_steps: int = 30
    num_trajectory_candidates: int = 6


class PlanningHead(nn.Module):
    """
    Planning head that outputs control actions.

    Takes encoded features and outputs steering + acceleration.
    Supports both deterministic and stochastic action output.
    """

    def __init__(self, input_dim: int, config: ModularPolicyConfig):
        super().__init__()
        self.config = config

        # Actor MLP
        layers = []
        in_dim = input_dim

        for hidden_dim in config.actor_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.action_dim))
        self.actor_mlp = nn.Sequential(*layers)

        # Action scaling parameters
        accel_center = (config.accel_range[0] + config.accel_range[1]) / 2
        accel_scale = (config.accel_range[1] - config.accel_range[0]) / 2
        steer_center = (config.steering_range[0] + config.steering_range[1]) / 2
        steer_scale = (config.steering_range[1] - config.steering_range[0]) / 2

        self.register_buffer('action_scale',
                             torch.tensor([steer_scale, accel_scale]))
        self.register_buffer('action_bias',
                             torch.tensor([steer_center, accel_center]))

        # Optional trajectory head
        if config.predict_trajectory:
            traj_output_dim = (
                config.num_trajectory_candidates *
                config.trajectory_steps * 3  # x, y, heading
            )
            self.trajectory_head = nn.Sequential(
                nn.Linear(input_dim, config.actor_hidden_dims[0]),
                get_activation(config.activation),
                nn.Linear(config.actor_hidden_dims[0], traj_output_dim),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(input_dim, 128),
                get_activation(config.activation),
                nn.Linear(128, config.num_trajectory_candidates),
            )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through planning head.

        Args:
            features: [B, input_dim] encoded features

        Returns:
            dict with:
                'action': [B, 2] scaled action
                'action_raw': [B, 2] pre-tanh action
                'steering': [B, 1]
                'acceleration': [B, 1]
        """
        action_raw = self.actor_mlp(features)
        action_scaled = torch.tanh(action_raw) * self.action_scale + self.action_bias

        output = {
            'action': action_scaled,
            'action_raw': action_raw,
            'steering': action_scaled[:, 0:1],
            'acceleration': action_scaled[:, 1:2],
        }

        # Trajectory prediction
        if self.config.predict_trajectory:
            B = features.shape[0]
            traj_raw = self.trajectory_head(features)
            trajectories = traj_raw.view(
                B, self.config.num_trajectory_candidates,
                self.config.trajectory_steps, 3
            )
            confidence = torch.softmax(self.confidence_head(features), dim=-1)
            weighted_traj = (trajectories * confidence.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

            output['trajectories'] = trajectories
            output['confidence'] = confidence
            output['trajectory'] = weighted_traj

        return output


class ModularDrivingPolicy(nn.Module):
    """
    Actor-Critic policy with modular encoder for autonomous driving.

    Supports:
    - Freeze/unfreeze individual encoder modules
    - Add new encoder modules with weight preservation
    - Partial checkpoint loading
    - ONNX export for Unity Sentis inference
    """

    def __init__(self, config: ModularPolicyConfig):
        super().__init__()
        self.config = config

        # Modular encoder
        self.encoder = ModularEncoder(config.encoder_config)

        # Planning head (actor)
        self.planner = PlanningHead(
            input_dim=self.encoder.output_dim,
            config=config,
        )

        # Value head (critic)
        critic_layers = []
        in_dim = self.encoder.output_dim

        for hidden_dim in config.critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(get_activation(config.activation))
            in_dim = hidden_dim

        critic_layers.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*critic_layers)

        # Learnable log standard deviation for stochastic policy
        self.log_std = nn.Parameter(
            torch.ones(config.action_dim) * config.log_std_init
        )

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            obs: [B, total_obs_dim] observation vector

        Returns:
            dict with action, value, and optional trajectory
        """
        features = self.encoder(obs)
        output = self.planner(features)
        output['value'] = self.value_head(features)
        return output

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value for PPO rollout.

        Args:
            obs: [B, total_obs_dim] observation
            deterministic: If True, return mean action

        Returns:
            action: [B, action_dim] sampled action
            log_prob: [B, 1] log probability
            value: [B, 1] state value
        """
        features = self.encoder(obs)
        planner_out = self.planner(features)
        value = self.value_head(features)

        action_mean = planner_out['action']
        log_std = torch.clamp(
            self.log_std, self.config.log_std_min, self.config.log_std_max
        )
        action_std = log_std.exp().expand_as(action_mean)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], 1, device=obs.device)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: [B, total_obs_dim] observations
            actions: [B, action_dim] actions to evaluate

        Returns:
            log_prob: [B, 1]
            entropy: [B, 1]
            value: [B, 1]
        """
        features = self.encoder(obs)
        planner_out = self.planner(features)
        value = self.value_head(features)

        action_mean = planner_out['action']
        log_std = torch.clamp(
            self.log_std, self.config.log_std_min, self.config.log_std_max
        )
        action_std = log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action for inference (no gradient)"""
        with torch.no_grad():
            action, _, _ = self.get_action_and_value(obs, deterministic=deterministic)
            return action

    # ========== Modular Encoder Methods ==========

    def freeze_encoder(self, name: str):
        """Freeze a specific encoder module"""
        self.encoder.freeze_encoder(name)

    def unfreeze_encoder(self, name: str):
        """Unfreeze a specific encoder module"""
        self.encoder.unfreeze_encoder(name)

    def freeze_all_encoders(self):
        """Freeze all encoder modules"""
        self.encoder.freeze_all_encoders()

    def unfreeze_all_encoders(self):
        """Unfreeze all encoder modules"""
        self.encoder.unfreeze_all_encoders()

    def add_encoder(
        self,
        enc_config: EncoderModuleConfig,
        freeze_existing: bool = True,
        trainable_fusion: bool = True
    ):
        """
        Add a new encoder module to the policy.

        Args:
            enc_config: Configuration for new encoder
            freeze_existing: Freeze existing encoders to preserve weights
            trainable_fusion: Keep fusion layer trainable
        """
        self.encoder.add_encoder(
            enc_config,
            freeze_existing=freeze_existing,
            trainable_fusion=trainable_fusion,
        )
        # Update config
        self.config.encoder_config = self.encoder.config

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Get all trainable parameters (excluding frozen encoders).

        Use this to create optimizer with only trainable params:
            optimizer = Adam(policy.get_trainable_params(), lr=1.5e-4)
        """
        params = []

        # Encoder params (filtered by frozen status)
        params.extend(self.encoder.get_trainable_params())

        # Planner params (always trainable unless explicitly frozen)
        params.extend(self.planner.parameters())

        # Value head params
        params.extend(self.value_head.parameters())

        # Log std
        params.append(self.log_std)

        return params

    def get_encoder_status(self) -> Dict[str, Dict]:
        """Get status of all encoder modules"""
        return self.encoder.get_encoder_status()

    # ========== Checkpoint Loading ==========

    def load_encoder_weights(
        self,
        checkpoint_path: str,
        encoder_names: Optional[List[str]] = None,
        strict: bool = False
    ) -> List[str]:
        """
        Load encoder weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            encoder_names: List of encoder names to load (None = all available)
            strict: If True, raise error on missing keys

        Returns:
            List of encoder names that were loaded
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip 'encoder.' prefix if present (from policy state_dict)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key[len('encoder.'):]] = value
            elif key.startswith('encoder_modules.') or key.startswith('fusion.'):
                encoder_state_dict[key] = value

        return self.encoder.load_encoder_weights(encoder_state_dict, encoder_names, strict)

    def load_fusion_weights(
        self,
        checkpoint_path: str,
        match_input_dim: bool = True
    ) -> bool:
        """
        Load fusion layer weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            match_input_dim: Only load if input dimensions match

        Returns:
            True if weights were loaded successfully
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip 'encoder.' prefix if present (from policy state_dict)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key[len('encoder.'):]] = value
            elif key.startswith('fusion.'):
                encoder_state_dict[key] = value

        return self.encoder.load_fusion_weights(encoder_state_dict, match_input_dim)

    def load_head_weights(
        self,
        checkpoint_path: str,
        load_planner: bool = True,
        load_value: bool = True,
        load_log_std: bool = True
    ) -> Dict[str, bool]:
        """
        Load planner/value head weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_planner: Whether to load planner weights
            load_value: Whether to load value head weights
            load_log_std: Whether to load log_std parameter

        Returns:
            Dict indicating which components were loaded
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        result = {'planner': False, 'value': False, 'log_std': False}

        # Load planner
        if load_planner:
            planner_state = {}
            for key, value in state_dict.items():
                if key.startswith('planner.'):
                    planner_state[key[len('planner.'):]] = value
            if planner_state:
                try:
                    self.planner.load_state_dict(planner_state, strict=False)
                    result['planner'] = True
                except RuntimeError:
                    pass

        # Load value head
        if load_value:
            value_state = {}
            for key, value in state_dict.items():
                if key.startswith('value_head.'):
                    value_state[key[len('value_head.'):]] = value
            if value_state:
                try:
                    self.value_head.load_state_dict(value_state, strict=False)
                    result['value'] = True
                except RuntimeError:
                    pass

        # Load log_std
        if load_log_std and 'log_std' in state_dict:
            if state_dict['log_std'].shape == self.log_std.shape:
                self.log_std.data = state_dict['log_std']
                result['log_std'] = True

        return result

    # ========== ONNX Export ==========

    def export_onnx(
        self,
        filepath: str,
        batch_size: int = 1,
        opset_version: int = 17
    ):
        """
        Export policy to ONNX format for Unity Sentis.

        Args:
            filepath: Output ONNX file path
            batch_size: Batch size for dummy input
            opset_version: ONNX opset version (17 recommended for Sentis)
        """
        self.eval()

        # Create wrapper that outputs only action components
        class ONNXWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs):
                action = self.policy.get_action(obs, deterministic=True)
                steering = action[:, 0:1]
                acceleration = action[:, 1:2]
                return steering, acceleration

        wrapper = ONNXWrapper(self)

        # Create dummy input
        total_obs_dim = self.encoder.total_input_dim
        dummy_input = torch.randn(batch_size, total_obs_dim)

        # Export
        torch.onnx.export(
            wrapper,
            dummy_input,
            filepath,
            input_names=['observation'],
            output_names=['steering', 'acceleration'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'steering': {0: 'batch_size'},
                'acceleration': {0: 'batch_size'},
            },
            opset_version=opset_version,
        )
        print(f"Model exported to {filepath}")
        print(f"  Input: observation [{batch_size}, {total_obs_dim}]")
        print(f"  Outputs: steering [{batch_size}, 1], acceleration [{batch_size}, 1]")

    # ========== Properties ==========

    @property
    def total_obs_dim(self) -> int:
        """Total observation dimension"""
        return self.encoder.total_input_dim

    @property
    def num_parameters(self) -> int:
        """Total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters (excluding frozen)"""
        return sum(p.numel() for p in self.get_trainable_params())


def create_modular_policy_config_phase_b() -> ModularPolicyConfig:
    """Create ModularPolicyConfig for Phase B (242D observations)"""
    from .modular_encoder import create_phase_b_config

    return ModularPolicyConfig(
        encoder_config=create_phase_b_config(),
        action_dim=2,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 128],
        predict_trajectory=False,
    )


def create_modular_policy_config_phase_c1() -> ModularPolicyConfig:
    """
    Create ModularPolicyConfig for Phase C-1 (254D observations).

    Includes lane encoder (12D) on top of Phase B.
    """
    from .modular_encoder import create_phase_b_config, create_lane_encoder_config

    encoder_config = create_phase_b_config()

    # Add lane encoder
    lane_config = create_lane_encoder_config()
    encoder_config.encoders[lane_config.name] = lane_config
    encoder_config.encoder_order.append(lane_config.name)

    # Update fusion to handle new dimensions
    # Total encoder output: 64 + 64 + 128 + 64 + 32 + 32 = 384

    return ModularPolicyConfig(
        encoder_config=encoder_config,
        action_dim=2,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 128],
        predict_trajectory=False,
    )
