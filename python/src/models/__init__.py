"""
Models Module - E2E Autonomous Driving (Tesla FSD Style)

Architecture:
  Level 1: MLP Planner (238D vector → control)
  Level 2: + CNN Backbone (images → features → control)
  Level 3: + BEV + Occupancy (spatial understanding)
  Level 4: + Temporal Fusion (full E2E)

Submodules:
- backbone: Feature extraction (RegNet, ResNet)
- neck: Multi-scale fusion (BiFPN)
- perception: Occupancy Network, BEV Former
- temporal: Temporal fusion (LSTM, Transformer)
- planning: Planning Network, Policy, Reward
- e2e_model: Unified E2E model
- modular_encoder: Modular Encoder Architecture for incremental learning
- modular_policy: Modular Driving Policy with freeze/unfreeze capability
"""

from .e2e_model import (
    E2EDrivingModel,
    E2EDrivingModelRL,
    E2EModelConfig,
)

from .modular_encoder import (
    EncoderModuleConfig,
    ModularEncoderConfig,
    ModularEncoder,
    create_phase_b_config,
    create_lane_encoder_config,
)

from .modular_policy import (
    ModularDrivingPolicy,
    ModularPolicyConfig,
    create_modular_policy_config_phase_b,
    create_modular_policy_config_phase_c1,
)

__all__ = [
    # E2E Model
    "E2EDrivingModel",
    "E2EDrivingModelRL",
    "E2EModelConfig",
    # Modular Encoder
    "EncoderModuleConfig",
    "ModularEncoderConfig",
    "ModularEncoder",
    "create_phase_b_config",
    "create_lane_encoder_config",
    # Modular Policy
    "ModularDrivingPolicy",
    "ModularPolicyConfig",
    "create_modular_policy_config_phase_b",
    "create_modular_policy_config_phase_c1",
    # Submodules
    "backbone",
    "neck",
    "perception",
    "temporal",
    "planning",
]
