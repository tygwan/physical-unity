"""
Policy Networks for Planning Module

Actor-Critic architecture for PPO/SAC algorithms.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional

from .encoder import ObservationEncoder


class PlanningPolicy(nn.Module):
    """
    Actor-Critic Policy for Motion Planning

    Actor outputs: [acceleration, steering] (continuous)
    Critic outputs: state value V(s)

    Action space:
        - acceleration: [-4.0, 2.0] m/s²
        - steering: [-0.5, 0.5] rad
    """

    def __init__(
        self,
        ego_dim: int = 8,
        route_dim: int = 30,
        surr_dim: int = 40,
        hidden_dim: int = 256,
        action_dim: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared encoder
        self.encoder = ObservationEncoder(
            ego_dim=ego_dim,
            route_dim=route_dim,
            surr_dim=surr_dim,
            hidden_dim=hidden_dim
        )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action scaling
        self.action_scale = torch.tensor([3.0, 0.5])  # [acc_range/2, steer_range]
        self.action_bias = torch.tensor([-1.0, 0.0])  # [acc_center, steer_center]

    def forward(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy

        Returns:
            action_mean: [batch, action_dim]
            action_std: [batch, action_dim]
            value: [batch, 1]
        """
        encoded = self.encoder(ego_state, route_info, surrounding)

        # Actor
        actor_hidden = self.actor(encoded)
        action_mean = self.action_mean(actor_hidden)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        action_std = action_log_std.exp()

        # Critic
        value = self.critic(encoded)

        return action_mean, action_std, value

    def get_action(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        Args:
            ego_state, route_info, surrounding: Observation tensors
            deterministic: If True, return mean action

        Returns:
            action: [batch, action_dim] - scaled to actual action space
            log_prob: [batch, 1] - log probability of action
            value: [batch, 1] - state value
        """
        action_mean, action_std, value = self.forward(
            ego_state, route_info, surrounding
        )

        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action[:, :1])
        else:
            dist = Normal(action_mean, action_std)
            action = dist.rsample()  # reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Scale action to actual range
        scaled_action = self._scale_action(action)

        return scaled_action, log_prob, value

    def evaluate_actions(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update

        Returns:
            log_prob: [batch, 1]
            entropy: [batch, 1]
            value: [batch, 1]
        """
        action_mean, action_std, value = self.forward(
            ego_state, route_info, surrounding
        )

        # Unscale actions
        unscaled_actions = self._unscale_action(actions)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(unscaled_actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to actual range"""
        action = torch.tanh(action)  # Squash to [-1, 1]
        return action * self.action_scale.to(action.device) + self.action_bias.to(action.device)

    def _unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Unscale action from actual range to [-1, 1]"""
        action = (action - self.action_bias.to(action.device)) / self.action_scale.to(action.device)
        # Inverse tanh (atanh)
        action = torch.clamp(action, -0.999, 0.999)
        return torch.atanh(action)


class BCPolicy(nn.Module):
    """
    Policy for Behavioral Cloning (supervised learning)

    Simpler architecture without value function.
    """

    def __init__(
        self,
        ego_dim: int = 8,
        route_dim: int = 30,
        surr_dim: int = 40,
        hidden_dim: int = 256,
        action_dim: int = 2
    ):
        super().__init__()

        self.encoder = ObservationEncoder(
            ego_dim=ego_dim,
            route_dim=route_dim,
            surr_dim=surr_dim,
            hidden_dim=hidden_dim
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Action scaling
        self.action_scale = torch.tensor([3.0, 0.5])
        self.action_bias = torch.tensor([-1.0, 0.0])

    def forward(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Returns:
            action: [batch, action_dim] - scaled to actual action space
        """
        encoded = self.encoder(ego_state, route_info, surrounding)
        action = self.policy(encoded)
        action = torch.tanh(action)
        return action * self.action_scale.to(action.device) + self.action_bias.to(action.device)
