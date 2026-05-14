"""DualBranchDecoder — fuse a spatial CNN branch with a spectral CNN branch.

Combines the spatial inductive bias of an ImageNet-pretrained ResNet-18 with
a frequency-domain branch (the same FFT → CNN stack as ``SpectralDecoder``).
The two branches are pooled independently and concatenated; the spectral
branch is guaranteed to see the whole image at every layer (every FFT bin
aggregates over every pixel), while the spatial branch keeps locality for
fine texture cues.

Hypothesis: bits 3 / 4 / 5 (bright/dark, contrast, saturation) — the bits
the spatial-only baseline struggles with — should pick up most of their
signal from the spectral branch's low-frequency bins, while bits 1 / 2 / 6
(sharpness, grain, detail) should ride on the spatial branch's local
features. Test by comparing per-bit accuracy against the single-branch
baselines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

if __package__:
    from .spectral import ConvBlock, fft_log_magnitude, IMAGENET_MEAN, IMAGENET_STD
else:  # allow `python3 dual_branch.py` for the __main__ smoke block
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spectral import ConvBlock, fft_log_magnitude, IMAGENET_MEAN, IMAGENET_STD


class DualBranchDecoder(nn.Module):
    NUM_BITS = 8

    SPATIAL_BACKBONES = {
        "resnet18": (lambda w: models.resnet18(weights=w), "ResNet18_Weights"),
        "resnet34": (lambda w: models.resnet34(weights=w), "ResNet34_Weights"),
        "resnet50": (lambda w: models.resnet50(weights=w), "ResNet50_Weights"),
    }

    def __init__(
        self,
        num_outputs: int = 8,
        pretrained: bool = True,
        spatial_backbone: str = "resnet18",
        fft_size: int = 256,
        base_ch: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.spatial_backbone_name = spatial_backbone

        # Spatial branch: configurable ResNet trunk + adaptive avg pool. The
        # heavier backbones (R-34, R-50) trade FLOPs for additional spatial
        # capacity on the bits where pure-frequency features plateau (the
        # ablation showed bits 3, 4 - bright/dark, contrast - get a 4-5pp
        # lift from the spatial branch even at R-18).
        if spatial_backbone not in self.SPATIAL_BACKBONES:
            raise ValueError(
                f"unknown spatial_backbone {spatial_backbone!r}; "
                f"expected one of {sorted(self.SPATIAL_BACKBONES)}"
            )
        ctor, weights_name = self.SPATIAL_BACKBONES[spatial_backbone]
        weights_cls = getattr(models, weights_name)
        weights = weights_cls.DEFAULT if pretrained else None
        sb = ctor(weights)
        self.spatial_dim = sb.fc.in_features  # 512 for R-18/34, 2048 for R-50
        self.spatial = nn.Sequential(
            sb.conv1, sb.bn1, sb.relu, sb.maxpool,
            sb.layer1, sb.layer2, sb.layer3, sb.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Spectral branch.
        self.register_buffer("imnet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imnet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.spec_downsample = nn.AdaptiveAvgPool2d(fft_size)
        self.spec_encoder = nn.Sequential(
            ConvBlock(3, base_ch, stride=2),
            ConvBlock(base_ch, base_ch * 2, stride=2),
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),
            ConvBlock(base_ch * 4, base_ch * 8, stride=2),
            ConvBlock(base_ch * 8, base_ch * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.spec_dim = base_ch * 8

        self.fusion = nn.Sequential(
            nn.LayerNorm(self.spatial_dim + self.spec_dim),
            nn.Linear(self.spatial_dim + self.spec_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_outputs),
        )

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        img = (x * self.imnet_std + self.imnet_mean).clamp(0.0, 1.0)
        img = self.spec_downsample(img)
        spec = fft_log_magnitude(img)
        mu = spec.mean(dim=(-1, -2), keepdim=True)
        sigma = spec.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return (spec - mu) / sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.spatial(x)
        spectral_feat = self.spec_encoder(self._spectrum(x))
        return self.fusion(torch.cat([spatial_feat, spectral_feat], dim=-1))

    def forward_with_branch_weights(
        self,
        x: torch.Tensor,
        spatial_weight: float = 1.0,
        spectral_weight: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with per-branch feature scaling before fusion.

        Both branches are *always* evaluated (so spectrum normalization and
        spatial pool statistics are unchanged); the pooled feature vector is
        scaled by ``spatial_weight`` / ``spectral_weight`` before being fed
        to ``fusion``. Identical to ``forward`` when both weights are 1.0
        (verified up to floating-point identity).

        Setting a weight to 0.0 reproduces the ablation modes in
        ``scripts/ablate_dual_branch.py`` (no_spectral / no_spatial). Setting
        a weight in (0, 1) attenuates that branch; values >1 amplify it.
        """
        spatial_feat = self.spatial(x) * spatial_weight
        spectral_feat = self.spec_encoder(self._spectrum(x)) * spectral_weight
        return self.fusion(torch.cat([spatial_feat, spectral_feat], dim=-1))


if __name__ == "__main__":
    model = DualBranchDecoder(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.2f}M")
