"""
End-to-End Driving Model (Tesla FSD Style)

Supports multiple complexity levels:
  Level 1: MLP Planner (238D vector → control)
  Level 2: + CNN Backbone (images → features → control)
  Level 3: + BEV + Occupancy (spatial understanding)
  Level 4: + Temporal Fusion (full E2E)

Current implementation: Level 1 (MLP) + Level 2 (CNN) ready
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class E2EModelConfig:
    """Configuration for E2E Driving Model"""

    # Level selection
    level: int = 1  # 1=MLP, 2=CNN, 3=BEV, 4=Full

    # Observation dimensions (Level 1)
    ego_dim: int = 8        # x, y, vx, vy, cos_h, sin_h, ax, ay
    ego_history_dim: int = 40   # 5 past steps x 8D
    agent_dim: int = 160    # 20 agents x 8 features
    route_dim: int = 30     # 10 waypoints x 3 (x, y, dist)
    total_obs_dim: int = 238  # sum of above

    # Action space
    action_dim: int = 2     # steering, acceleration
    steering_range: Tuple[float, float] = (-0.5, 0.5)    # radians
    accel_range: Tuple[float, float] = (-4.0, 2.0)       # m/s^2

    # Network architecture
    hidden_dims: list = field(default_factory=lambda: [512, 512, 256])
    activation: str = "relu"
    dropout: float = 0.1

    # Trajectory prediction (auxiliary)
    predict_trajectory: bool = True
    trajectory_steps: int = 30    # 3 seconds @ 10Hz
    num_trajectory_candidates: int = 6

    # Camera config (Level 2+)
    num_cameras: int = 8
    image_size: Tuple[int, int] = (224, 224)  # H, W
    backbone: str = "resnet18"  # resnet18, resnet50, regnet

    # BEV config (Level 3+)
    bev_size: int = 100     # 100x100 grid
    bev_resolution: float = 1.0  # 1m per cell
    bev_channels: int = 256

    # Temporal config (Level 4)
    num_frames: int = 10
    temporal_method: str = "transformer"  # lstm, transformer, conv3d


def get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    return activations.get(name, nn.ReLU())


class MLPEncoder(nn.Module):
    """Level 1: MLP-based observation encoder for 238D vector input"""

    def __init__(self, config: E2EModelConfig):
        super().__init__()
        self.config = config

        # Separate encoders for each observation component
        self.ego_encoder = nn.Sequential(
            nn.Linear(config.ego_dim, 64),
            get_activation(config.activation),
            nn.Linear(64, 64),
            get_activation(config.activation),
        )

        self.history_encoder = nn.Sequential(
            nn.Linear(config.ego_history_dim, 128),
            get_activation(config.activation),
            nn.Linear(128, 64),
            get_activation(config.activation),
        )

        self.agent_encoder = nn.Sequential(
            nn.Linear(config.agent_dim, 256),
            get_activation(config.activation),
            nn.Linear(256, 128),
            get_activation(config.activation),
        )

        self.route_encoder = nn.Sequential(
            nn.Linear(config.route_dim, 64),
            get_activation(config.activation),
            nn.Linear(64, 64),
            get_activation(config.activation),
        )

        # Fusion layer
        fusion_dim = 64 + 64 + 128 + 64  # 320
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dims[0]),
            get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[0]),
            get_activation(config.activation),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, 238] concatenated observation vector

        Returns:
            features: [B, hidden_dim]
        """
        # Split observation into components
        ego = obs[:, :self.config.ego_dim]
        history = obs[:, self.config.ego_dim:self.config.ego_dim + self.config.ego_history_dim]
        agents = obs[:, self.config.ego_dim + self.config.ego_history_dim:
                      self.config.ego_dim + self.config.ego_history_dim + self.config.agent_dim]
        route = obs[:, -self.config.route_dim:]

        # Encode each component
        ego_feat = self.ego_encoder(ego)
        hist_feat = self.history_encoder(history)
        agent_feat = self.agent_encoder(agents)
        route_feat = self.route_encoder(route)

        # Fuse
        combined = torch.cat([ego_feat, hist_feat, agent_feat, route_feat], dim=-1)
        return self.fusion(combined)


class AttentionAgentEncoder(nn.Module):
    """Attention-based encoder for surrounding agents (variable count)"""

    def __init__(self, agent_feature_dim: int = 8, max_agents: int = 20,
                 hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.agent_feature_dim = agent_feature_dim
        self.max_agents = max_agents

        self.agent_embed = nn.Sequential(
            nn.Linear(agent_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, agents_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agents_flat: [B, max_agents * agent_feature_dim]

        Returns:
            features: [B, hidden_dim]
        """
        B = agents_flat.shape[0]
        agents = agents_flat.view(B, self.max_agents, self.agent_feature_dim)

        # Embed each agent
        agent_embeds = self.agent_embed(agents)  # [B, N, hidden]

        # Self-attention
        attended, _ = self.self_attention(agent_embeds, agent_embeds, agent_embeds)

        # Pool (mean)
        pooled = attended.mean(dim=1)  # [B, hidden]
        return self.output_proj(pooled)


class PlanningHead(nn.Module):
    """Planning network: features → control actions + trajectory"""

    def __init__(self, config: E2EModelConfig):
        super().__init__()
        self.config = config

        input_dim = config.hidden_dims[0]

        # Control output (steering + acceleration)
        self.control_head = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[1]),
            get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            get_activation(config.activation),
            nn.Linear(config.hidden_dims[2], config.action_dim),
        )

        # Trajectory prediction (auxiliary task)
        if config.predict_trajectory:
            self.trajectory_head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dims[1]),
                get_activation(config.activation),
                nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
                get_activation(config.activation),
                nn.Linear(config.hidden_dims[2],
                          config.num_trajectory_candidates * config.trajectory_steps * 3),
            )

            # Trajectory confidence scoring
            self.confidence_head = nn.Sequential(
                nn.Linear(input_dim, 128),
                get_activation(config.activation),
                nn.Linear(128, config.num_trajectory_candidates),
            )

        # Action scaling parameters
        accel_center = (config.accel_range[0] + config.accel_range[1]) / 2  # -1.0
        accel_scale = (config.accel_range[1] - config.accel_range[0]) / 2   # 3.0
        steer_center = (config.steering_range[0] + config.steering_range[1]) / 2  # 0.0
        steer_scale = (config.steering_range[1] - config.steering_range[0]) / 2   # 0.5

        self.register_buffer('action_scale',
                             torch.tensor([steer_scale, accel_scale]))
        self.register_buffer('action_bias',
                             torch.tensor([steer_center, accel_center]))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, hidden_dim]

        Returns:
            dict with:
                'action': [B, 2] (steering, acceleration) - scaled
                'action_raw': [B, 2] (pre-tanh values for loss)
                'trajectory': [B, num_candidates, steps, 3] (optional)
                'confidence': [B, num_candidates] (optional)
        """
        # Control output
        action_raw = self.control_head(features)
        action_scaled = torch.tanh(action_raw) * self.action_scale + self.action_bias

        output = {
            'action': action_scaled,
            'action_raw': action_raw,
            'steering': action_scaled[:, 0:1],
            'acceleration': action_scaled[:, 1:2],
        }

        # Trajectory prediction
        if self.config.predict_trajectory:
            traj_raw = self.trajectory_head(features)
            B = features.shape[0]
            trajectories = traj_raw.view(
                B, self.config.num_trajectory_candidates,
                self.config.trajectory_steps, 3
            )
            confidence = torch.softmax(self.confidence_head(features), dim=-1)

            # Weighted trajectory (best prediction)
            weighted_traj = (trajectories * confidence.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

            output['trajectories'] = trajectories
            output['confidence'] = confidence
            output['trajectory'] = weighted_traj  # [B, steps, 3]

        return output


class E2EDrivingModel(nn.Module):
    """
    Tesla-style End-to-End Driving Model

    Level 1 (Current): 238D observation → MLP → steering + acceleration
    Level 2 (Future): Camera images → CNN → features → control
    Level 3 (Future): + BEV + Occupancy
    Level 4 (Future): + Temporal Fusion

    Training modes:
        - BC (Behavioral Cloning): Supervised learning from expert
        - RL (PPO/SAC): Reinforcement learning in simulation
        - Hybrid: BC pre-train → RL fine-tune
    """

    def __init__(self, config: Optional[E2EModelConfig] = None):
        super().__init__()
        self.config = config or E2EModelConfig()

        # Build encoder based on level
        if self.config.level == 1:
            self.encoder = MLPEncoder(self.config)
        elif self.config.level == 2:
            from .backbone.resnet import CNNEncoder
            self.encoder = CNNEncoder(self.config)
        else:
            raise NotImplementedError(f"Level {self.config.level} not yet implemented")

        # Planning head (shared across levels)
        self.planner = PlanningHead(self.config)

    def forward(self, obs: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: [B, obs_dim] observation vector (Level 1)
                 OR [B, obs_dim] vector obs for fusion (Level 2)
            images: [B, num_cameras, 3, H, W] camera images (Level 2+)

        Returns:
            Dict with action, trajectory, confidence
        """
        if self.config.level == 1:
            features = self.encoder(obs)
        elif self.config.level >= 2:
            if images is None:
                raise ValueError("Level 2+ requires 'images' input")
            features = self.encoder(images, vector_obs=obs)
        else:
            features = self.encoder(obs)
        return self.planner(features)

    def get_action(self, obs: torch.Tensor,
                   images: Optional[torch.Tensor] = None,
                   deterministic: bool = True) -> torch.Tensor:
        """
        Get action for inference (no gradient)

        Returns:
            action: [B, 2] (steering, acceleration)
        """
        with torch.no_grad():
            output = self.forward(obs, images=images)
            return output['action']

    def compute_bc_loss(self, obs: torch.Tensor,
                        expert_action: torch.Tensor,
                        expert_trajectory: Optional[torch.Tensor] = None,
                        images: Optional[torch.Tensor] = None,
                        ) -> Dict[str, torch.Tensor]:
        """
        Compute Behavioral Cloning loss

        Args:
            obs: [B, obs_dim]
            expert_action: [B, 2] (steering, acceleration)
            expert_trajectory: [B, steps, 3] (optional)
            images: [B, num_cameras, 3, H, W] (Level 2+)

        Returns:
            Dict with total_loss and component losses
        """
        output = self.forward(obs, images=images)

        # Action loss (MSE on scaled actions)
        action_loss = nn.functional.mse_loss(output['action'], expert_action)

        # Separate steering and acceleration losses for monitoring
        steer_loss = nn.functional.mse_loss(
            output['steering'], expert_action[:, 0:1])
        accel_loss = nn.functional.mse_loss(
            output['acceleration'], expert_action[:, 1:2])

        total_loss = action_loss

        losses = {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'steering_loss': steer_loss,
            'acceleration_loss': accel_loss,
        }

        # Trajectory loss (auxiliary)
        if expert_trajectory is not None and 'trajectory' in output:
            traj_loss = nn.functional.mse_loss(
                output['trajectory'][:, :expert_trajectory.shape[1], :],
                expert_trajectory
            )
            losses['trajectory_loss'] = traj_loss
            total_loss = total_loss + 0.2 * traj_loss
            losses['total_loss'] = total_loss

        return losses

    def export_onnx(self, filepath: str, batch_size: int = 1):
        """Export model to ONNX format for Unity Sentis"""
        self.eval()
        dummy_input = torch.randn(batch_size, self.config.total_obs_dim)

        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            input_names=['observation'],
            output_names=['steering', 'acceleration'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'steering': {0: 'batch_size'},
                'acceleration': {0: 'batch_size'},
            },
            opset_version=17,
        )
        print(f"Model exported to {filepath}")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class E2EDrivingModelRL(E2EDrivingModel):
    """
    RL-compatible version with Actor-Critic architecture

    Adds:
        - Value function (critic)
        - Stochastic action sampling
        - Log probability computation
    """

    def __init__(self, config: Optional[E2EModelConfig] = None):
        super().__init__(config)

        # Value function (critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
            get_activation(self.config.activation),
            nn.Linear(self.config.hidden_dims[1], 128),
            get_activation(self.config.activation),
            nn.Linear(128, 1),
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.config.action_dim))

    def forward(self, obs: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        if self.config.level == 1:
            features = self.encoder(obs)
        elif self.config.level >= 2:
            if images is None:
                raise ValueError("Level 2+ requires 'images' input")
            features = self.encoder(images, vector_obs=obs)
        else:
            features = self.encoder(obs)
        output = self.planner(features)
        output['value'] = self.value_head(features)
        return output

    def _encode(self, obs: torch.Tensor,
                images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared encoder logic for RL methods"""
        if self.config.level == 1:
            return self.encoder(obs)
        elif self.config.level >= 2:
            if images is None:
                raise ValueError("Level 2+ requires 'images' input")
            return self.encoder(images, vector_obs=obs)
        return self.encoder(obs)

    def get_action_and_value(self, obs: torch.Tensor,
                             images: Optional[torch.Tensor] = None,
                             deterministic: bool = False
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value for PPO

        Returns:
            action: [B, 2]
            log_prob: [B, 1]
            value: [B, 1]
        """
        features = self._encode(obs, images)
        planner_out = self.planner(features)
        value = self.value_head(features)

        action_mean = planner_out['action']
        action_std = self.log_std.exp().expand_as(action_mean)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], 1, device=obs.device)
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         images: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update

        Returns:
            log_prob: [B, 1]
            entropy: [B, 1]
            value: [B, 1]
        """
        features = self._encode(obs, images)
        planner_out = self.planner(features)
        value = self.value_head(features)

        action_mean = planner_out['action']
        action_std = self.log_std.exp().expand_as(action_mean)

        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value
