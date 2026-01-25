"""
Modular Encoder Architecture for Autonomous Driving

Enables incremental learning by preserving training when observation space changes.
Based on research from Progressive Neural Networks, UPGD (ICLR 2024), and EWC.

Key Features:
- Named encoder modules for different observation groups
- Freeze/unfreeze capability for selective training
- Dynamic encoder addition without losing learned weights
- Fusion layer expansion with weight transfer

Example usage:
    # Create initial encoder with Phase B observations (242D)
    config = ModularEncoderConfig(encoders={
        "ego": EncoderModuleConfig("ego", 8, [64], 64),
        "history": EncoderModuleConfig("history", 40, [128], 64),
        "agents": EncoderModuleConfig("agents", 160, [256], 128),
        "route": EncoderModuleConfig("route", 30, [64], 64),
        "speed": EncoderModuleConfig("speed", 4, [32], 32),
    })
    encoder = ModularEncoder(config)

    # Later, add lane encoder (12D) for Phase C-1
    lane_config = EncoderModuleConfig("lane", 12, [32, 32], 32)
    encoder.add_encoder(lane_config, freeze_existing=True)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, OrderedDict as OrderedDictType
from collections import OrderedDict
import copy


@dataclass
class EncoderModuleConfig:
    """Configuration for a single encoder module"""
    name: str                           # e.g., "ego", "lane", "history"
    input_dim: int                      # Input dimension for this module
    hidden_dims: List[int]              # Hidden layer sizes
    output_dim: int                     # Output feature dimension
    frozen: bool = False                # Whether weights are frozen
    activation: str = "relu"            # Activation function
    dropout: float = 0.0                # Dropout rate (0 = no dropout)

    def __post_init__(self):
        if not self.hidden_dims:
            self.hidden_dims = [self.output_dim]


@dataclass
class ModularEncoderConfig:
    """Configuration for the complete modular encoder"""
    encoders: Dict[str, EncoderModuleConfig] = field(default_factory=dict)
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    fusion_output_dim: int = 512
    fusion_activation: str = "relu"
    fusion_dropout: float = 0.1

    # Observation slicing order (important for consistent forward pass)
    encoder_order: Optional[List[str]] = None

    def __post_init__(self):
        # If encoder_order not specified, use insertion order
        if self.encoder_order is None:
            self.encoder_order = list(self.encoders.keys())

    @property
    def total_input_dim(self) -> int:
        """Total input dimension across all encoders"""
        return sum(enc.input_dim for enc in self.encoders.values())

    @property
    def total_encoder_output_dim(self) -> int:
        """Total encoder output dimension (fusion layer input)"""
        return sum(enc.output_dim for enc in self.encoders.values())


def get_activation(name: str) -> nn.Module:
    """Get activation function by name"""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }
    return activations.get(name, nn.ReLU())


class EncoderModule(nn.Module):
    """
    Single encoder module for a specific observation group.

    Converts input observations to a fixed-size feature vector.
    Supports freezing/unfreezing for selective training.
    """

    def __init__(self, config: EncoderModuleConfig):
        super().__init__()
        self.config = config
        self.name = config.name

        # Build MLP layers
        layers = []
        in_dim = config.input_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Final projection to output_dim if needed
        if config.hidden_dims[-1] != config.output_dim:
            layers.append(nn.Linear(config.hidden_dims[-1], config.output_dim))
            layers.append(get_activation(config.activation))

        self.mlp = nn.Sequential(*layers)

        # Apply freezing if configured
        if config.frozen:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        return self.mlp(x)

    def freeze(self):
        """Freeze all parameters in this module"""
        for param in self.parameters():
            param.requires_grad = False
        self.config.frozen = True

    def unfreeze(self):
        """Unfreeze all parameters in this module"""
        for param in self.parameters():
            param.requires_grad = True
        self.config.frozen = False

    @property
    def is_frozen(self) -> bool:
        """Check if module is frozen"""
        return self.config.frozen


class FusionLayer(nn.Module):
    """
    Fusion layer that combines outputs from all encoder modules.

    Supports weight transfer when expanding for new encoders.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, activation: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(get_activation(activation))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModularEncoder(nn.Module):
    """
    Modular Encoder with named encoder modules and fusion layer.

    Enables:
    - Freeze/unfreeze individual encoders
    - Add new encoders while preserving existing weights
    - Partial checkpoint loading (load only compatible encoders)
    """

    def __init__(self, config: ModularEncoderConfig):
        super().__init__()
        self.config = config

        # Build encoder modules (use ModuleDict for named access)
        self.encoder_modules = nn.ModuleDict()
        for name, enc_config in config.encoders.items():
            self.encoder_modules[name] = EncoderModule(enc_config)

        # Compute slice indices for observation splitting
        self._slice_indices = self._compute_slice_indices()

        # Build fusion layer
        self.fusion = self._build_fusion()

    def _compute_slice_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Compute observation slice indices for each encoder.
        Returns dict mapping encoder name to (start_idx, end_idx).
        """
        indices = {}
        current_idx = 0

        for name in self.config.encoder_order:
            if name in self.config.encoders:
                input_dim = self.config.encoders[name].input_dim
                indices[name] = (current_idx, current_idx + input_dim)
                current_idx += input_dim

        return indices

    def _build_fusion(self) -> FusionLayer:
        """Build fusion layer with current total encoder output dimension"""
        total_encoder_out = sum(
            self.config.encoders[name].output_dim
            for name in self.config.encoder_order
            if name in self.config.encoders
        )
        return FusionLayer(
            input_dim=total_encoder_out,
            hidden_dims=self.config.fusion_hidden_dims,
            output_dim=self.config.fusion_output_dim,
            activation=self.config.fusion_activation,
            dropout=self.config.fusion_dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all encoders and fusion layer.

        Args:
            obs: [B, total_input_dim] concatenated observation vector

        Returns:
            features: [B, fusion_output_dim]
        """
        # Encode each observation group
        encoder_outputs = []

        for name in self.config.encoder_order:
            if name not in self._slice_indices:
                continue

            start_idx, end_idx = self._slice_indices[name]
            obs_slice = obs[:, start_idx:end_idx]
            encoded = self.encoder_modules[name](obs_slice)
            encoder_outputs.append(encoded)

        # Concatenate all encoder outputs
        combined = torch.cat(encoder_outputs, dim=-1)

        # Fuse
        features = self.fusion(combined)
        return features

    def freeze_encoder(self, name: str):
        """Freeze a specific encoder by name"""
        if name in self.encoder_modules:
            self.encoder_modules[name].freeze()
        else:
            raise ValueError(f"Encoder '{name}' not found. Available: {list(self.encoder_modules.keys())}")

    def unfreeze_encoder(self, name: str):
        """Unfreeze a specific encoder by name"""
        if name in self.encoder_modules:
            self.encoder_modules[name].unfreeze()
        else:
            raise ValueError(f"Encoder '{name}' not found. Available: {list(self.encoder_modules.keys())}")

    def freeze_all_encoders(self):
        """Freeze all encoder modules"""
        for name in self.encoder_modules:
            self.encoder_modules[name].freeze()

    def unfreeze_all_encoders(self):
        """Unfreeze all encoder modules"""
        for name in self.encoder_modules:
            self.encoder_modules[name].unfreeze()

    def add_encoder(self, enc_config: EncoderModuleConfig,
                    freeze_existing: bool = True,
                    trainable_fusion: bool = True):
        """
        Add a new encoder module and expand fusion layer.

        This is the key method for incremental learning:
        1. Optionally freeze existing encoders to preserve learned features
        2. Add new encoder module
        3. Expand fusion layer with weight transfer

        Args:
            enc_config: Configuration for new encoder
            freeze_existing: If True, freeze all existing encoders
            trainable_fusion: If True, keep fusion layer trainable
        """
        if enc_config.name in self.encoder_modules:
            raise ValueError(f"Encoder '{enc_config.name}' already exists")

        # 1. Freeze existing encoders if requested
        if freeze_existing:
            self.freeze_all_encoders()

        # 2. Add new encoder module
        new_encoder = EncoderModule(enc_config)
        self.encoder_modules[enc_config.name] = new_encoder
        self.config.encoders[enc_config.name] = enc_config
        self.config.encoder_order.append(enc_config.name)

        # 3. Update slice indices
        self._slice_indices = self._compute_slice_indices()

        # 4. Expand fusion layer with weight transfer
        old_fusion = self.fusion
        self.fusion = self._build_fusion()
        self._transfer_fusion_weights(old_fusion)

        # 5. Optionally freeze fusion layer
        if not trainable_fusion:
            for param in self.fusion.parameters():
                param.requires_grad = False

    def _transfer_fusion_weights(self, old_fusion: FusionLayer):
        """
        Transfer weights from old fusion layer to new expanded fusion.

        Strategy:
        - Copy weights for existing encoder dimensions
        - Initialize new dimensions with small random values
        """
        old_input_dim = old_fusion.input_dim
        new_input_dim = self.fusion.input_dim

        if old_input_dim == new_input_dim:
            # Same size, just copy
            self.fusion.load_state_dict(old_fusion.state_dict())
            return

        # Get first layer weights
        old_state = old_fusion.state_dict()
        new_state = self.fusion.state_dict()

        for key in old_state:
            if key.endswith('.weight') and 'mlp.0' in key:
                # First layer weight: [out_features, in_features]
                old_weight = old_state[key]
                new_weight = new_state[key]

                # Copy old weights to corresponding positions
                new_weight[:, :old_input_dim] = old_weight

                # Initialize new dimensions with small random values
                if new_input_dim > old_input_dim:
                    nn.init.xavier_uniform_(
                        new_weight[:, old_input_dim:].unsqueeze(0)
                    )
                    # Scale down to not disrupt learned features
                    new_weight[:, old_input_dim:] *= 0.1

                new_state[key] = new_weight

            elif key.endswith('.bias') and 'mlp.0' in key:
                # First layer bias: copy as-is
                new_state[key] = old_state[key]

            elif key in old_state and key in new_state:
                # Other layers: copy if same shape
                if old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]

        self.fusion.load_state_dict(new_state)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable (non-frozen) parameters"""
        params = []
        for encoder in self.encoder_modules.values():
            if not encoder.is_frozen:
                params.extend(encoder.parameters())
        params.extend(self.fusion.parameters())
        return params

    def get_encoder_status(self) -> Dict[str, Dict]:
        """Get status of all encoders (for logging/debugging)"""
        status = {}
        for name, encoder in self.encoder_modules.items():
            config = encoder.config
            num_params = sum(p.numel() for p in encoder.parameters())
            trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            status[name] = {
                'input_dim': config.input_dim,
                'output_dim': config.output_dim,
                'hidden_dims': config.hidden_dims,
                'frozen': config.frozen,
                'num_params': num_params,
                'trainable_params': trainable,
            }
        return status

    @property
    def total_input_dim(self) -> int:
        """Total input observation dimension"""
        return self.config.total_input_dim

    @property
    def output_dim(self) -> int:
        """Output feature dimension (fusion output)"""
        return self.config.fusion_output_dim

    @property
    def num_encoders(self) -> int:
        """Number of encoder modules"""
        return len(self.encoder_modules)

    def load_encoder_weights(self, state_dict: Dict, encoder_names: Optional[List[str]] = None,
                              strict: bool = False) -> List[str]:
        """
        Load encoder weights from a state dict (partial loading supported).

        Args:
            state_dict: Model state dict containing encoder weights
            encoder_names: List of encoder names to load (None = all available)
            strict: If True, raise error on missing keys

        Returns:
            List of encoder names that were loaded
        """
        loaded = []
        encoder_names = encoder_names or list(self.encoder_modules.keys())

        for name in encoder_names:
            if name not in self.encoder_modules:
                if strict:
                    raise KeyError(f"Encoder '{name}' not in current model")
                continue

            # Find matching keys in state_dict
            prefix = f"encoder_modules.{name}."
            encoder_state = {}

            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    encoder_state[new_key] = value

            if encoder_state:
                try:
                    self.encoder_modules[name].load_state_dict(encoder_state, strict=strict)
                    loaded.append(name)
                except RuntimeError as e:
                    if strict:
                        raise
                    print(f"Warning: Could not load encoder '{name}': {e}")

        return loaded

    def load_fusion_weights(self, state_dict: Dict, match_input_dim: bool = True) -> bool:
        """
        Load fusion layer weights from a state dict.

        Args:
            state_dict: Model state dict containing fusion weights
            match_input_dim: If True, only load if input dimensions match

        Returns:
            True if weights were loaded successfully
        """
        # Find fusion keys
        fusion_state = {}
        for key, value in state_dict.items():
            if key.startswith("fusion."):
                new_key = key[len("fusion."):]
                fusion_state[new_key] = value

        if not fusion_state:
            return False

        # Check input dimension compatibility
        if match_input_dim:
            first_layer_key = "mlp.0.weight"
            if first_layer_key in fusion_state:
                old_in_dim = fusion_state[first_layer_key].shape[1]
                new_in_dim = self.fusion.input_dim
                if old_in_dim != new_in_dim:
                    print(f"Fusion input dim mismatch: {old_in_dim} vs {new_in_dim}")
                    return False

        try:
            self.fusion.load_state_dict(fusion_state, strict=True)
            return True
        except RuntimeError as e:
            print(f"Warning: Could not load fusion weights: {e}")
            return False


def create_phase_b_config() -> ModularEncoderConfig:
    """
    Create ModularEncoderConfig for Phase B (242D observations).

    This matches the observation space from Phase B training:
    - ego: 8D (x, y, vx, vy, cos_h, sin_h, ax, ay)
    - history: 40D (5 past steps x 8D)
    - agents: 160D (20 agents x 8 features)
    - route: 30D (10 waypoints x 3: x, y, dist)
    - speed: 4D (speed-related features)
    """
    return ModularEncoderConfig(
        encoders={
            "ego": EncoderModuleConfig(
                name="ego",
                input_dim=8,
                hidden_dims=[64],
                output_dim=64,
            ),
            "history": EncoderModuleConfig(
                name="history",
                input_dim=40,
                hidden_dims=[128],
                output_dim=64,
            ),
            "agents": EncoderModuleConfig(
                name="agents",
                input_dim=160,
                hidden_dims=[256],
                output_dim=128,
            ),
            "route": EncoderModuleConfig(
                name="route",
                input_dim=30,
                hidden_dims=[64],
                output_dim=64,
            ),
            "speed": EncoderModuleConfig(
                name="speed",
                input_dim=4,
                hidden_dims=[32],
                output_dim=32,
            ),
        },
        encoder_order=["ego", "history", "agents", "route", "speed"],
        fusion_hidden_dims=[512, 512],
        fusion_output_dim=512,
    )


def create_lane_encoder_config() -> EncoderModuleConfig:
    """
    Create EncoderModuleConfig for lane information (12D).

    Lane observation:
    - left_lane: 4D (distance, angle, curvature, width)
    - right_lane: 4D (distance, angle, curvature, width)
    - current_lane: 4D (offset, heading_error, curvature, width)
    """
    return EncoderModuleConfig(
        name="lane",
        input_dim=12,
        hidden_dims=[32, 32],
        output_dim=32,
        frozen=False,
    )
