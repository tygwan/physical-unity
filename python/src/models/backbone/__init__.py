"""
Backbone networks for visual feature extraction.

Supports:
- ResNet-18/34/50 (pretrained on ImageNet)
- Multi-camera fusion (concat_mlp, attention, avg_pool)
- Freezing strategies for transfer learning
"""

from .resnet import (
    ResNetBackbone,
    MultiCameraFusion,
    CNNEncoder,
    BackboneConfig,
)

__all__ = [
    "ResNetBackbone",
    "MultiCameraFusion",
    "CNNEncoder",
    "BackboneConfig",
]
