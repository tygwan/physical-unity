"""
ResNet-based Backbone for Camera Feature Extraction

Supports multi-camera input (Tesla-style 8 cameras).
Extracts visual features from each camera independently,
then fuses them into a unified representation.

Architecture:
    [B, N_cam, 3, H, W] → ResNet-18 (shared) → [B, N_cam, feat_dim]
    → Multi-camera Fusion → [B, fused_dim]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """Configuration for CNN backbone"""
    backbone_type: str = "resnet18"  # resnet18, resnet34, resnet50
    pretrained: bool = True
    num_cameras: int = 8
    image_size: Tuple[int, int] = (224, 224)
    feature_dim: int = 512  # ResNet-18/34=512, ResNet-50=2048
    fusion_method: str = "concat_mlp"  # concat_mlp, attention, avg_pool
    fused_dim: int = 512
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Freeze first N layers (0=none, 4=all for resnet)


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for single-camera feature extraction.
    Uses torchvision's pretrained ResNet with final FC removed.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        # Build backbone
        self._build_backbone()

        # Optionally freeze layers
        if config.freeze_backbone:
            self._freeze_all()
        elif config.freeze_layers > 0:
            self._freeze_n_layers(config.freeze_layers)

    def _build_backbone(self):
        """Build ResNet backbone from torchvision"""
        try:
            import torchvision.models as models
            weights_map = {
                "resnet18": ("resnet18", models.ResNet18_Weights.DEFAULT if self.config.pretrained else None),
                "resnet34": ("resnet34", models.ResNet34_Weights.DEFAULT if self.config.pretrained else None),
                "resnet50": ("resnet50", models.ResNet50_Weights.DEFAULT if self.config.pretrained else None),
            }

            model_name, weights = weights_map.get(
                self.config.backbone_type,
                ("resnet18", models.ResNet18_Weights.DEFAULT if self.config.pretrained else None)
            )

            backbone = getattr(models, model_name)(weights=weights)
        except ImportError:
            # Fallback: build minimal ResNet-18 without torchvision
            backbone = self._build_minimal_resnet18()

        # Remove final FC and avgpool - we'll handle pooling ourselves
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Update feature dim based on actual backbone
        if self.config.backbone_type in ("resnet18", "resnet34"):
            self._actual_feat_dim = 512
        else:  # resnet50
            self._actual_feat_dim = 2048

        # Project to desired feature dim if different
        if self._actual_feat_dim != self.config.feature_dim:
            self.proj = nn.Linear(self._actual_feat_dim, self.config.feature_dim)
        else:
            self.proj = nn.Identity()

    def _build_minimal_resnet18(self):
        """Minimal ResNet-18 without torchvision dependency"""
        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, in_ch, out_ch, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.downsample = downsample

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample:
                    identity = self.downsample(x)
                return self.relu(out + identity)

        class MinimalResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, 2, 1)

                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)

            def _make_layer(self, in_ch, out_ch, blocks, stride=1):
                downsample = None
                if stride != 1 or in_ch != out_ch:
                    downsample = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )
                layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
                for _ in range(1, blocks):
                    layers.append(BasicBlock(out_ch, out_ch))
                return nn.Sequential(*layers)

        return MinimalResNet()

    def _freeze_all(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def _freeze_n_layers(self, n: int):
        """Freeze first n layers (layer0=conv1+bn1, layer1-4=residual blocks)"""
        layers = [
            nn.Sequential(self.features[0], self.features[1]),  # conv1+bn1
            self.features[4],  # layer1
            self.features[5],  # layer2
            self.features[6],  # layer3
            self.features[7],  # layer4
        ]
        for i in range(min(n, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single camera image.

        Args:
            x: [B, 3, H, W] RGB image

        Returns:
            features: [B, feature_dim]
        """
        feat = self.features(x)         # [B, C, h, w]
        feat = self.pool(feat)          # [B, C, 1, 1]
        feat = feat.flatten(1)          # [B, C]
        feat = self.proj(feat)          # [B, feature_dim]
        return feat


class MultiCameraFusion(nn.Module):
    """
    Fuses features from multiple cameras into a single representation.

    Methods:
        concat_mlp: Concatenate all camera features + MLP projection
        attention: Cross-camera attention pooling
        avg_pool: Simple average pooling
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        if config.fusion_method == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(config.feature_dim * config.num_cameras, config.fused_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.fused_dim * 2, config.fused_dim),
                nn.ReLU(),
            )
        elif config.fusion_method == "attention":
            self.query = nn.Linear(config.feature_dim, config.fused_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=config.fused_dim,
                num_heads=8,
                batch_first=True,
            )
            self.output_proj = nn.Linear(config.fused_dim, config.fused_dim)
        elif config.fusion_method == "avg_pool":
            self.fusion = nn.Sequential(
                nn.Linear(config.feature_dim, config.fused_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown fusion method: {config.fusion_method}")

    def forward(self, camera_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-camera features.

        Args:
            camera_features: [B, num_cameras, feature_dim]

        Returns:
            fused: [B, fused_dim]
        """
        if self.config.fusion_method == "concat_mlp":
            flat = camera_features.flatten(1)  # [B, num_cameras * feature_dim]
            return self.fusion(flat)

        elif self.config.fusion_method == "attention":
            q = self.query(camera_features)  # [B, N, fused_dim]
            attended, _ = self.attention(q, q, q)  # [B, N, fused_dim]
            pooled = attended.mean(dim=1)  # [B, fused_dim]
            return self.output_proj(pooled)

        elif self.config.fusion_method == "avg_pool":
            avg = camera_features.mean(dim=1)  # [B, feature_dim]
            return self.fusion(avg)


class CNNEncoder(nn.Module):
    """
    Level 2 Encoder: Multi-camera CNN + optional vector observation fusion.

    Pipeline:
        Camera images → Shared ResNet → Multi-camera Fusion → [fused_dim]
        Vector obs → MLP → [vector_feat_dim]
        Concat → Fusion MLP → [output_dim]
    """

    def __init__(self, config: 'E2EModelConfig'):
        super().__init__()
        self.config = config

        # Backbone config
        bb_config = BackboneConfig(
            backbone_type=config.backbone,
            pretrained=True,
            num_cameras=config.num_cameras,
            image_size=config.image_size,
            feature_dim=512,
            fusion_method="concat_mlp" if config.num_cameras <= 8 else "attention",
            fused_dim=config.hidden_dims[0],
        )

        # Shared backbone (applied per camera)
        self.backbone = ResNetBackbone(bb_config)

        # Multi-camera fusion
        self.camera_fusion = MultiCameraFusion(bb_config)

        # Optional: vector observation encoder (for hybrid Level 1+2)
        self.use_vector_obs = config.total_obs_dim > 0
        if self.use_vector_obs:
            self.vector_encoder = nn.Sequential(
                nn.Linear(config.total_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            # Final fusion of camera + vector features
            self.final_fusion = nn.Sequential(
                nn.Linear(config.hidden_dims[0] + 128, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.final_fusion = nn.Identity()

    def forward(self, images: torch.Tensor,
                vector_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: [B, num_cameras, 3, H, W] multi-camera input
            vector_obs: [B, obs_dim] optional vector observations

        Returns:
            features: [B, hidden_dim]
        """
        B, N, C, H, W = images.shape

        # Process each camera through shared backbone
        images_flat = images.view(B * N, C, H, W)
        cam_features = self.backbone(images_flat)  # [B*N, feat_dim]
        cam_features = cam_features.view(B, N, -1)  # [B, N, feat_dim]

        # Fuse cameras
        fused = self.camera_fusion(cam_features)  # [B, hidden_dim]

        # Fuse with vector observations if available
        if self.use_vector_obs and vector_obs is not None:
            vec_feat = self.vector_encoder(vector_obs)  # [B, 128]
            combined = torch.cat([fused, vec_feat], dim=-1)
            fused = self.final_fusion(combined)  # [B, hidden_dim]

        return fused
