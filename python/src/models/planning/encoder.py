"""
Observation Encoder for Planning Module

Encodes multi-modal observations (ego state, route, surrounding vehicles)
into a latent representation for the policy network.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ObservationEncoder(nn.Module):
    """
    Multi-modal observation encoder for autonomous driving planning

    Input dimensions (default):
        - ego_state: [batch, 8] (x, y, vx, vy, cos_h, sin_h, ax, ay)
        - route_info: [batch, 30] (10 waypoints x 3: x, y, dist)
        - surrounding: [batch, 40] (8 vehicles x 5: x, y, vx, vy, heading)

    Output:
        - encoded: [batch, hidden_dim] (default 256)
    """

    def __init__(
        self,
        ego_dim: int = 8,
        route_dim: int = 30,
        surr_dim: int = 40,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.ego_dim = ego_dim
        self.route_dim = route_dim
        self.surr_dim = surr_dim
        self.hidden_dim = hidden_dim

        # Ego state encoder
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Route encoder
        self.route_encoder = nn.Sequential(
            nn.Linear(route_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Surrounding vehicles encoder
        self.surr_encoder = nn.Sequential(
            nn.Linear(surr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Fusion layer
        fusion_dim = 64 + 64 + 128  # ego + route + surrounding
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode observations into latent representation

        Args:
            ego_state: [batch, ego_dim]
            route_info: [batch, route_dim]
            surrounding: [batch, surr_dim]

        Returns:
            encoded: [batch, hidden_dim]
        """
        ego_feat = self.ego_encoder(ego_state)
        route_feat = self.route_encoder(route_info)
        surr_feat = self.surr_encoder(surrounding)

        combined = torch.cat([ego_feat, route_feat, surr_feat], dim=-1)
        encoded = self.fusion(combined)

        return encoded

    def get_input_dim(self) -> int:
        """Return total input dimension"""
        return self.ego_dim + self.route_dim + self.surr_dim

    def get_output_dim(self) -> int:
        """Return output dimension"""
        return self.hidden_dim


class AttentionObservationEncoder(nn.Module):
    """
    Observation encoder with attention mechanism for surrounding vehicles

    Uses self-attention to handle variable numbers of surrounding vehicles.
    """

    def __init__(
        self,
        ego_dim: int = 8,
        route_dim: int = 30,
        vehicle_dim: int = 5,
        max_vehicles: int = 8,
        hidden_dim: int = 256,
        n_heads: int = 4
    ):
        super().__init__()

        self.vehicle_dim = vehicle_dim
        self.max_vehicles = max_vehicles

        # Ego state encoder
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Route encoder
        self.route_encoder = nn.Sequential(
            nn.Linear(route_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Vehicle encoder (per-vehicle)
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Self-attention for vehicles
        self.vehicle_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=n_heads,
            batch_first=True
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        ego_state: torch.Tensor,
        route_info: torch.Tensor,
        surrounding: torch.Tensor  # [batch, max_vehicles, vehicle_dim]
    ) -> torch.Tensor:
        batch_size = ego_state.shape[0]

        # Encode ego and route
        ego_feat = self.ego_encoder(ego_state)
        route_feat = self.route_encoder(route_info)

        # Encode each vehicle
        vehicle_feat = self.vehicle_encoder(
            surrounding.view(-1, self.vehicle_dim)
        ).view(batch_size, self.max_vehicles, -1)

        # Apply self-attention
        attended, _ = self.vehicle_attention(
            vehicle_feat, vehicle_feat, vehicle_feat
        )

        # Pool attended features (mean pooling)
        surr_feat = attended.mean(dim=1)

        # Fuse all features
        combined = torch.cat([ego_feat, route_feat, surr_feat], dim=-1)
        encoded = self.fusion(combined)

        return encoded
