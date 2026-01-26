"""
Hybrid Policy for Phase C-1 Training

This module implements a progressive learning approach that:
1. Freezes Phase B encoder (242D → 512D) to preserve learned knowledge (+903 reward)
2. Adds new lane encoder (12D → 32D) for Phase C features
3. Creates a combining layer (544D → 512D) to merge both representations

Architecture:
                    ┌─────────────────────────┐
    242D ──────────►│  Phase B Encoder (FROZEN) │──────────►512D─┐
    (ego, history,  │  (3 layers x 512)        │                 │
     agents, route, └─────────────────────────┘                 │
     speed)                                                      ▼
                                                          ┌────────────┐
                    ┌─────────────────────────┐           │  Combiner  │──►512D──►Policy/Value
    12D ───────────►│  Lane Encoder (TRAINABLE)│──────────►32D─┤  (544→512) │
    (lane info)     │  (2 layers x 32)         │              └────────────┘
                    └─────────────────────────┘

Usage:
    # Load Phase B checkpoint
    policy = HybridDrivingPolicy.from_phase_b_checkpoint(
        checkpoint_path='results/v12_phaseB/E2EDrivingAgent/E2EDrivingAgent-2000150.pt',
        freeze_phase_b=True
    )

    # Forward pass with 254D observation
    obs = torch.randn(batch_size, 254)  # 242 + 12
    output = policy(obs)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class HybridPolicyConfig:
    """Configuration for HybridDrivingPolicy."""
    # Phase B encoder (frozen)
    phase_b_input_dim: int = 242
    phase_b_hidden_dims: List[int] = None  # Will be [512, 512, 512]
    phase_b_output_dim: int = 512

    # Lane encoder (trainable)
    lane_input_dim: int = 12
    lane_hidden_dims: List[int] = None  # Default: [32, 32]
    lane_output_dim: int = 32

    # Combiner (trainable)
    combiner_hidden_dim: int = 512
    combiner_output_dim: int = 512

    # Action space
    action_dim: int = 2  # steering, acceleration

    # Value head
    value_hidden_dim: int = 256

    def __post_init__(self):
        if self.phase_b_hidden_dims is None:
            self.phase_b_hidden_dims = [512, 512, 512]
        if self.lane_hidden_dims is None:
            self.lane_hidden_dims = [32, 32]


class PhaseBEncoder(nn.Module):
    """
    Frozen encoder from Phase B training (242D → 512D).
    Loads weights from ML-Agents checkpoint format.
    """

    def __init__(self, input_dim: int = 242, hidden_dims: List[int] = None, output_dim: int = 512):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build sequential layers (matching ML-Agents structure)
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SELU())  # ML-Agents uses SELU by default
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.is_frozen = False

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.is_frozen = True

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.is_frozen = False

    def load_from_mlagents(self, policy_state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Load weights from ML-Agents checkpoint Policy dict.

        Args:
            policy_state_dict: The 'Policy' dict from ML-Agents checkpoint

        Returns:
            True if successful
        """
        # ML-Agents key pattern: network_body._body_endoder.seq_layers.{idx}.{weight|bias}
        # Note: ML-Agents has a typo "endoder" instead of "encoder"

        layer_idx = 0
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                weight_key = f'network_body._body_endoder.seq_layers.{layer_idx}.weight'
                bias_key = f'network_body._body_endoder.seq_layers.{layer_idx}.bias'

                if weight_key in policy_state_dict and bias_key in policy_state_dict:
                    module.weight.data = policy_state_dict[weight_key].clone()
                    module.bias.data = policy_state_dict[bias_key].clone()
                    print(f"  Loaded layer {layer_idx}: {module.weight.shape}")
                else:
                    print(f"  Warning: Missing keys for layer {layer_idx}")

                layer_idx += 2  # Skip activation layer in ML-Agents numbering

        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LaneEncoder(nn.Module):
    """
    New encoder for lane observations (12D → 32D).
    Trainable during Phase C-1.
    """

    def __init__(self, input_dim: int = 12, hidden_dims: List[int] = None, output_dim: int = 32):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.SELU())

        self.layers = nn.Sequential(*layers)

        # Initialize with small weights to minimize disruption
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)  # Small gain
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FeatureCombiner(nn.Module):
    """
    Combines Phase B features (512D) with lane features (32D) → 512D.
    Uses residual-style connection to preserve Phase B information.
    """

    def __init__(self, phase_b_dim: int = 512, lane_dim: int = 32, output_dim: int = 512):
        super().__init__()

        self.phase_b_dim = phase_b_dim
        self.lane_dim = lane_dim
        self.output_dim = output_dim

        # Project lane features to match Phase B dimension
        self.lane_proj = nn.Sequential(
            nn.Linear(lane_dim, 128),
            nn.SELU(),
            nn.Linear(128, output_dim)
        )

        # Gating mechanism: learn how much to incorporate lane information
        self.gate = nn.Sequential(
            nn.Linear(phase_b_dim + lane_dim, 128),
            nn.SELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Initialize to preserve Phase B output initially
        self._init_weights()

    def _init_weights(self):
        # Initialize gate to output ~0.1 initially (small lane contribution)
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, -2.0)  # Sigmoid(-2) ≈ 0.12

        # Initialize lane projection with small weights
        for module in self.lane_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, phase_b_features: torch.Tensor, lane_features: torch.Tensor) -> torch.Tensor:
        # Compute lane contribution
        lane_proj = self.lane_proj(lane_features)

        # Compute gate (how much to incorporate lane info)
        combined_for_gate = torch.cat([phase_b_features, lane_features], dim=-1)
        gate_value = self.gate(combined_for_gate)

        # Residual combination: output = phase_b + gate * lane_proj
        output = phase_b_features + gate_value * lane_proj

        return output


class HybridDrivingPolicy(nn.Module):
    """
    Hybrid policy combining frozen Phase B encoder with trainable lane encoder.

    This preserves Phase B knowledge (+903 reward) while learning to use
    new lane information during Phase C-1 training.
    """

    def __init__(self, config: HybridPolicyConfig = None):
        super().__init__()

        if config is None:
            config = HybridPolicyConfig()

        self.config = config
        self.phase_b_input_dim = config.phase_b_input_dim
        self.lane_input_dim = config.lane_input_dim
        self.total_obs_dim = config.phase_b_input_dim + config.lane_input_dim

        # Phase B encoder (will be frozen after loading)
        self.phase_b_encoder = PhaseBEncoder(
            input_dim=config.phase_b_input_dim,
            hidden_dims=config.phase_b_hidden_dims,
            output_dim=config.phase_b_output_dim
        )

        # Lane encoder (trainable)
        self.lane_encoder = LaneEncoder(
            input_dim=config.lane_input_dim,
            hidden_dims=config.lane_hidden_dims,
            output_dim=config.lane_output_dim
        )

        # Feature combiner
        self.combiner = FeatureCombiner(
            phase_b_dim=config.phase_b_output_dim,
            lane_dim=config.lane_output_dim,
            output_dim=config.combiner_output_dim
        )

        # Policy head (Actor) - Match ML-Agents architecture (single layer 512→2)
        # This allows loading Phase B policy weights directly
        self.policy_head = nn.Linear(config.combiner_output_dim, config.action_dim)

        # Value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.combiner_output_dim, config.value_hidden_dim),
            nn.SELU(),
            nn.Linear(config.value_hidden_dim, 1)
        )

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(1, config.action_dim))

    def _split_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split 254D observation into Phase B (242D) and lane (12D) parts."""
        phase_b_obs = obs[..., :self.phase_b_input_dim]
        lane_obs = obs[..., self.phase_b_input_dim:]
        return phase_b_obs, lane_obs

    def encode(self, obs: torch.Tensor, bypass_combiner: bool = False) -> torch.Tensor:
        """
        Encode observation into combined features.

        Args:
            obs: (batch, 254) observation tensor
            bypass_combiner: If True, skip combiner and return Phase B features directly.
                            Use for debugging Phase B behavior.
        """
        phase_b_obs, lane_obs = self._split_observation(obs)

        # Encode Phase B observations (frozen)
        phase_b_features = self.phase_b_encoder(phase_b_obs)

        if bypass_combiner:
            # Skip combiner - return Phase B features directly
            return phase_b_features

        # Encode lane observations (trainable)
        lane_features = self.lane_encoder(lane_obs)

        # Combine features
        combined = self.combiner(phase_b_features, lane_features)

        return combined

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning action mean and value."""
        features = self.encode(obs, bypass_combiner=self.bypass_combiner)

        # Raw action (before tanh) - this is how ML-Agents works
        raw_action = self.policy_head(features)
        value = self.value_head(features)

        # For deterministic action, apply tanh
        # For stochastic, get_action_and_value will sample then apply tanh
        action_mean = torch.tanh(raw_action)

        return {
            'action': action_mean,
            'raw_action': raw_action,  # Store raw for proper TanhNormal sampling
            'value': value,
            'features': features
        }

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value.

        Matches ML-Agents behavior:
        - action_mean = tanh(raw_action) - bounded deterministic action
        - Sample from Normal(action_mean, std)
        - Store unclamped sample for consistent log_prob computation
        - Environment will clamp to [-1, 1]

        Returns:
            action: (batch, action_dim) - unclamped sample for training consistency
            log_prob: (batch, 1)
            value: (batch, 1)
        """
        output = self(obs)
        action_mean = output['action']  # After tanh, in [-1, 1]
        value = output['value']

        # Create normal distribution centered at tanh(raw_action)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)

        if deterministic:
            # Deterministic: use the mean directly
            action = action_mean
        else:
            # Stochastic: sample around the mean
            # Don't clamp here - store the raw sample for consistent log_prob
            action = dist.rsample()

        # Simple Gaussian log probability (no Jacobian correction needed)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.

        Actions are the unclamped samples stored during rollout.
        We compute their log_prob under the current policy distribution.

        Returns:
            log_prob: (batch, 1)
            entropy: (batch, 1)
            value: (batch, 1)
        """
        output = self(obs)
        action_mean = output['action']  # After tanh, in [-1, 1]
        value = output['value']

        action_std = torch.exp(self.log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)

        # Simple Gaussian log probability
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value

    def freeze_phase_b(self):
        """Freeze Phase B encoder."""
        self.phase_b_encoder.freeze()
        print(f"Frozen Phase B encoder ({sum(p.numel() for p in self.phase_b_encoder.parameters())} params)")

    def unfreeze_phase_b(self):
        """Unfreeze Phase B encoder for fine-tuning."""
        self.phase_b_encoder.unfreeze()
        print(f"Unfrozen Phase B encoder")

    def freeze_policy_head(self):
        """Freeze policy head and log_std to preserve Phase B behavior during value warmup."""
        for param in self.policy_head.parameters():
            param.requires_grad = False
        self.log_std.requires_grad = False
        self._policy_head_frozen = True
        frozen_params = sum(p.numel() for p in self.policy_head.parameters()) + self.log_std.numel()
        print(f"Frozen policy_head + log_std ({frozen_params} params)")

    def unfreeze_policy_head(self):
        """Unfreeze policy head and log_std for training."""
        for param in self.policy_head.parameters():
            param.requires_grad = True
        self.log_std.requires_grad = True
        self._policy_head_frozen = False
        print(f"Unfrozen policy_head + log_std")

    def freeze_combiner_and_lane(self):
        """Freeze combiner and lane encoder during value-only warmup."""
        for param in self.combiner.parameters():
            param.requires_grad = False
        for param in self.lane_encoder.parameters():
            param.requires_grad = False
        self._combiner_frozen = True
        self._lane_frozen = True
        self._combiner_gate_frozen = True
        frozen_params = sum(p.numel() for p in self.combiner.parameters()) + \
                        sum(p.numel() for p in self.lane_encoder.parameters())
        print(f"Frozen combiner + lane_encoder ({frozen_params} params)")

    def unfreeze_combiner_and_lane(self):
        """Unfreeze combiner and lane encoder."""
        for param in self.combiner.parameters():
            param.requires_grad = True
        for param in self.lane_encoder.parameters():
            param.requires_grad = True
        self._combiner_frozen = False
        self._lane_frozen = False
        self._combiner_gate_frozen = False
        print(f"Unfrozen combiner + lane_encoder")

    # ========== Gradual Unfreezing Methods (Fine-grained control) ==========

    def freeze_lane_encoder(self):
        """Freeze lane encoder only."""
        for param in self.lane_encoder.parameters():
            param.requires_grad = False
        self._lane_frozen = True
        frozen_params = sum(p.numel() for p in self.lane_encoder.parameters())
        print(f"Frozen lane_encoder ({frozen_params} params)")

    def unfreeze_lane_encoder(self):
        """Unfreeze lane encoder only."""
        for param in self.lane_encoder.parameters():
            param.requires_grad = True
        self._lane_frozen = False
        trainable = sum(p.numel() for p in self.lane_encoder.parameters())
        print(f"Unfrozen lane_encoder ({trainable} params)")

    def freeze_combiner_gate(self):
        """Freeze only the combiner gate (not lane_proj)."""
        for param in self.combiner.gate.parameters():
            param.requires_grad = False
        self._combiner_gate_frozen = True
        frozen_params = sum(p.numel() for p in self.combiner.gate.parameters())
        print(f"Frozen combiner.gate ({frozen_params} params)")

    def unfreeze_combiner_gate(self):
        """Unfreeze combiner gate."""
        for param in self.combiner.gate.parameters():
            param.requires_grad = True
        self._combiner_gate_frozen = False
        trainable = sum(p.numel() for p in self.combiner.gate.parameters())
        print(f"Unfrozen combiner.gate ({trainable} params)")

    def freeze_combiner_lane_proj(self):
        """Freeze combiner lane projection."""
        for param in self.combiner.lane_proj.parameters():
            param.requires_grad = False
        self._combiner_lane_proj_frozen = True
        frozen_params = sum(p.numel() for p in self.combiner.lane_proj.parameters())
        print(f"Frozen combiner.lane_proj ({frozen_params} params)")

    def unfreeze_combiner_lane_proj(self):
        """Unfreeze combiner lane projection."""
        for param in self.combiner.lane_proj.parameters():
            param.requires_grad = True
        self._combiner_lane_proj_frozen = False
        trainable = sum(p.numel() for p in self.combiner.lane_proj.parameters())
        print(f"Unfrozen combiner.lane_proj ({trainable} params)")

    def freeze_full_combiner(self):
        """Freeze entire combiner (gate + lane_proj)."""
        for param in self.combiner.parameters():
            param.requires_grad = False
        self._combiner_frozen = True
        self._combiner_gate_frozen = True
        self._combiner_lane_proj_frozen = True
        frozen_params = sum(p.numel() for p in self.combiner.parameters())
        print(f"Frozen full combiner ({frozen_params} params)")

    def unfreeze_full_combiner(self):
        """Unfreeze entire combiner."""
        for param in self.combiner.parameters():
            param.requires_grad = True
        self._combiner_frozen = False
        self._combiner_gate_frozen = False
        self._combiner_lane_proj_frozen = False
        trainable = sum(p.numel() for p in self.combiner.parameters())
        print(f"Unfrozen full combiner ({trainable} params)")

    @property
    def is_combiner_frozen(self) -> bool:
        """Check if combiner is frozen."""
        return getattr(self, '_combiner_frozen', False)

    @property
    def is_lane_frozen(self) -> bool:
        """Check if lane encoder is frozen."""
        return getattr(self, '_lane_frozen', False)

    @property
    def is_combiner_gate_frozen(self) -> bool:
        """Check if combiner gate is frozen."""
        return getattr(self, '_combiner_gate_frozen', False)

    @property
    def is_combiner_lane_proj_frozen(self) -> bool:
        """Check if combiner lane projection is frozen."""
        return getattr(self, '_combiner_lane_proj_frozen', False)

    def set_bypass_combiner(self, bypass: bool = True):
        """Enable/disable combiner bypass mode for debugging Phase B behavior."""
        self._bypass_combiner = bypass
        print(f"Combiner bypass mode: {'ON (pure Phase B)' if bypass else 'OFF (hybrid)'}")

    @property
    def bypass_combiner(self) -> bool:
        """Check if combiner bypass is enabled."""
        return getattr(self, '_bypass_combiner', False)

    @property
    def is_policy_head_frozen(self) -> bool:
        """Check if policy head is frozen."""
        return getattr(self, '_policy_head_frozen', False)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters (excludes frozen components)."""
        params = []

        # Lane encoder (only if not frozen)
        if not self.is_lane_frozen:
            params.extend(self.lane_encoder.parameters())

        # Combiner gate (only if not frozen)
        if not self.is_combiner_gate_frozen:
            params.extend(self.combiner.gate.parameters())

        # Combiner lane_proj (only if not frozen)
        if not self.is_combiner_lane_proj_frozen:
            params.extend(self.combiner.lane_proj.parameters())

        # Policy head (only if not frozen)
        if not self.is_policy_head_frozen:
            params.extend(self.policy_head.parameters())

        # Value head (always trainable)
        params.extend(self.value_head.parameters())

        # Log std (only if policy_head not frozen)
        if not self.is_policy_head_frozen:
            params.append(self.log_std)

        # Phase B encoder (only if not frozen)
        if not self.phase_b_encoder.is_frozen:
            params.extend(self.phase_b_encoder.parameters())

        return params

    @property
    def num_trainable_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())

    @property
    def num_total_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_param_summary(self) -> Dict[str, int]:
        """Get parameter count summary."""
        return {
            'phase_b_encoder': sum(p.numel() for p in self.phase_b_encoder.parameters()),
            'lane_encoder': sum(p.numel() for p in self.lane_encoder.parameters()),
            'combiner': sum(p.numel() for p in self.combiner.parameters()),
            'policy_head': sum(p.numel() for p in self.policy_head.parameters()),
            'value_head': sum(p.numel() for p in self.value_head.parameters()),
            'log_std': self.log_std.numel(),
            'total': self.num_total_params,
            'trainable': self.num_trainable_params
        }

    @classmethod
    def from_phase_b_checkpoint(
        cls,
        checkpoint_path: str,
        config: HybridPolicyConfig = None,
        freeze_phase_b: bool = True,
        freeze_policy_head: bool = True,
        freeze_combiner: bool = True
    ) -> 'HybridDrivingPolicy':
        """
        Create HybridDrivingPolicy and load Phase B encoder weights.

        Args:
            checkpoint_path: Path to ML-Agents .pt checkpoint
            config: Policy configuration
            freeze_phase_b: Whether to freeze Phase B encoder

        Returns:
            Initialized HybridDrivingPolicy
        """
        if config is None:
            config = HybridPolicyConfig()

        # Create policy
        policy = cls(config)

        # Load checkpoint
        print(f"Loading Phase B checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract Policy state dict
        if 'Policy' in checkpoint:
            policy_state_dict = checkpoint['Policy']
        else:
            raise ValueError("Checkpoint missing 'Policy' key")

        # Load Phase B encoder weights
        print("Loading Phase B encoder weights...")
        policy.phase_b_encoder.load_from_mlagents(policy_state_dict)

        # Load action log_std if available
        if 'action_model._continuous_distribution.log_sigma' in policy_state_dict:
            log_sigma = policy_state_dict['action_model._continuous_distribution.log_sigma']
            policy.log_std.data = log_sigma.squeeze(0)
            print(f"  Loaded log_std: {policy.log_std.data}")

        # Load policy head (action model) weights - CRITICAL for Phase B behavior!
        mu_weight_key = 'action_model._continuous_distribution.mu.weight'
        mu_bias_key = 'action_model._continuous_distribution.mu.bias'
        if mu_weight_key in policy_state_dict and mu_bias_key in policy_state_dict:
            policy.policy_head.weight.data = policy_state_dict[mu_weight_key].clone()
            policy.policy_head.bias.data = policy_state_dict[mu_bias_key].clone()
            print(f"  Loaded policy_head: {policy.policy_head.weight.shape}")
        else:
            print("  WARNING: Policy head weights not found in checkpoint!")

        # Freeze Phase B encoder if requested
        if freeze_phase_b:
            policy.freeze_phase_b()

        # Freeze policy head if requested (for value warmup phase)
        if freeze_policy_head:
            policy.freeze_policy_head()

        # Freeze combiner and lane encoder if requested (for value-only warmup)
        if freeze_combiner:
            policy.freeze_lane_encoder()
            policy.freeze_full_combiner()

        # Print parameter summary
        summary = policy.get_param_summary()
        print(f"\nParameter Summary:")
        for name, count in summary.items():
            print(f"  {name}: {count:,}")

        return policy

    def export_onnx(self, filepath: str, opset_version: int = 17):
        """Export to ONNX for Unity Sentis inference."""
        self.eval()

        # Move to CPU for ONNX export (ONNX export works best on CPU)
        self_cpu = self.cpu()

        # Create wrapper that outputs only action
        class ONNXWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs):
                output = self.policy(obs)
                return output['action']

        wrapper = ONNXWrapper(self_cpu)

        # Dummy input (on CPU)
        dummy_input = torch.randn(1, self.total_obs_dim)

        # Export
        torch.onnx.export(
            wrapper,
            dummy_input,
            filepath,
            opset_version=opset_version,
            input_names=['obs'],
            output_names=['action'],
            dynamic_axes={
                'obs': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )

        print(f"Exported ONNX model to: {filepath}")


def create_hybrid_policy_config_phase_c1() -> HybridPolicyConfig:
    """Create default config for Phase C-1 training."""
    return HybridPolicyConfig(
        phase_b_input_dim=242,
        phase_b_hidden_dims=[512, 512, 512],
        phase_b_output_dim=512,
        lane_input_dim=12,
        lane_hidden_dims=[32, 32],
        lane_output_dim=32,
        combiner_hidden_dim=512,
        combiner_output_dim=512,
        action_dim=2,
        value_hidden_dim=256
    )


if __name__ == '__main__':
    # Test
    print("Testing HybridDrivingPolicy...")

    config = create_hybrid_policy_config_phase_c1()
    policy = HybridDrivingPolicy(config)

    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 254)

    output = policy(obs)
    print(f"\nForward pass:")
    print(f"  Input: {obs.shape}")
    print(f"  Action: {output['action'].shape}")
    print(f"  Value: {output['value'].shape}")

    # Test action sampling
    action, log_prob, value = policy.get_action_and_value(obs)
    print(f"\nAction sampling:")
    print(f"  Action: {action.shape}")
    print(f"  Log prob: {log_prob.shape}")
    print(f"  Value: {value.shape}")

    print("\nTest passed!")
