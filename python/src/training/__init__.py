"""
Training Module - Training pipelines for AD models

Components:
- train_bc: Behavioral Cloning training
- train_rl: PPO/SAC training with Unity environment
- train_gail: GAIL training
- train_hybrid: Hybrid BC + RL training (CIMRL)
"""

__all__ = ["train_bc", "train_rl", "train_gail", "train_hybrid"]
