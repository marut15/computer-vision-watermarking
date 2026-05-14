"""MultiScalePyramidDecoder — image-pyramid ResNet with multi-pool fusion.

The encoder's perturbations live in different bands at different scales:
luminance / contrast / saturation shifts (bits 3, 4, 5) dominate the heavily
downsampled view because antialiasing suppresses local texture, exposing
global tone. Sharpness / grain / detail bits (1, 2, 6) require near-full
resolution to be visible. A single-resolution CNN has to trade these off; a
multi-scale pyramid sees both at once.

Architecture:

* A shared, ImageNet-pretrained ResNet-18 backbone (its low parameter count
  is intentional — three branches' worth of ResNet-50 would be overkill at
  this dataset scale, and the existing 188 M-param 8 × ResNet-50 ensemble
  already shows that capacity per bit is not the bottleneck).
* The image is fed through the same backbone at three scales (1.0, 0.5,
  0.25 by default). Three scales × three pooling heads (avg, std, max) →
  9 × 512-dim per-branch features → 4608-dim fused vector → MLP → 8 logits.
* The std and max pools are deliberate. Standard global-average pooling
  collapses bit-relevant variance information; std pool gives the model an
  explicit handle on contrast bits, max pool exposes peak intensities, and
  avg pool keeps the standard global-average behaviour.

This is the most "broad-pattern" of the four new decoders — every branch
sees the entire image, and the only locality signal is what the ResNet
itself preserves.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiScalePyramidDecoder(nn.Module):
    NUM_BITS = 8

    def __init__(
        self,
        num_outputs: int = 8,
        scales: tuple[float, ...] = (1.0, 0.5, 0.25),
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.scales = tuple(scales)

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feat_dim = backbone.fc.in_features  # 512

        # Drop the avg-pool + fc; keep the conv trunk.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # 3 pools × len(scales) × feat_dim
        n_pooled = 3 * len(self.scales) * self.feat_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(n_pooled),
            nn.Linear(n_pooled, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    @staticmethod
    def _multipool(feat: torch.Tensor) -> torch.Tensor:
        """avg / std / max pool over spatial dims, concatenated channel-wise."""
        avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        mx = F.adaptive_max_pool2d(feat, 1).flatten(1)
        # Std-pool: per-channel std over spatial dims, biased estimator (matches BN convention).
        var = ((feat - avg.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-1, -2))
        std = var.clamp_min(1e-6).sqrt()
        return torch.cat([avg, std, mx], dim=-1)

    def _resize(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return x
        H, W = x.shape[-2:]
        # Keep at least 32×32 so the ResNet-18 trunk can still produce a non-empty feature map.
        new_h = max(32, int(round(H * scale)))
        new_w = max(32, int(round(W * scale)))
        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = []
        for s in self.scales:
            feat = self.stem(self._resize(x, s))
            pooled.append(self._multipool(feat))
        return self.fusion(torch.cat(pooled, dim=-1))


if __name__ == "__main__":
    model = MultiScalePyramidDecoder(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.2f}M")
