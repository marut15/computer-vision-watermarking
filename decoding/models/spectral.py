"""SpectralDecoder — operate on the FFT magnitude of the image.

Watermark perturbations from LoRA style-sliders modify the image's spectral
content (see ``fft_analysis`` in ``scripts/signal_analysis.py``). A CNN
trained in the spatial domain has to re-derive those spectral statistics
through dozens of local conv layers; a CNN trained in the frequency domain
reads them off directly.

Each FFT bin already aggregates over every spatial pixel, so by construction
this decoder's receptive field is the entire image at the very first layer.
There is no "single-point" attention to overcome: the input *is* a global
descriptor.

Pipeline:

1. Denormalise the ImageNet-normalised input back to [0, 1].
2. Adaptive-average-pool to ``fft_size`` × ``fft_size`` so the FFT is cheap
   regardless of the source resolution. The LoRA fingerprint dominates
   mid-frequencies, which are well-resolved at 256.
3. 2D FFT, take ``log1p(|.|)``, fftshift so DC is centred.
4. Per-image standardise the spectrum so absolute energy doesn't dominate.
5. Small CNN → adaptive avg pool → MLP → 8 logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fft_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` → centred log-magnitude of the 2D FFT, same shape."""
    fft = torch.fft.fft2(x)
    mag = torch.fft.fftshift(fft.abs(), dim=(-1, -2))
    return torch.log1p(mag)


class ConvBlock(nn.Module):
    """Conv → BN → GELU → Conv → BN → GELU, with optional spatial downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralDecoder(nn.Module):
    NUM_BITS = 8

    def __init__(
        self,
        num_outputs: int = 8,
        base_ch: int = 32,
        fft_size: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.downsample = nn.AdaptiveAvgPool2d(fft_size)
        self.register_buffer("imnet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imnet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.encoder = nn.Sequential(
            ConvBlock(3, base_ch, stride=2),               # 256 → 128
            ConvBlock(base_ch, base_ch * 2, stride=2),     # 128 → 64
            ConvBlock(base_ch * 2, base_ch * 4, stride=2), # 64 → 32
            ConvBlock(base_ch * 4, base_ch * 8, stride=2), # 32 → 16
            ConvBlock(base_ch * 8, base_ch * 8, stride=2), # 16 → 8
        )
        self.feat_dim = base_ch * 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, base_ch * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, num_outputs),
        )

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        img = (x * self.imnet_std + self.imnet_mean).clamp(0.0, 1.0)
        img = self.downsample(img)
        spec = fft_log_magnitude(img)
        # Per-image, per-channel standardise so different image means don't dominate.
        mu = spec.mean(dim=(-1, -2), keepdim=True)
        sigma = spec.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        return (spec - mu) / sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self._spectrum(x)
        feat = self.encoder(spec)
        return self.head(feat)


if __name__ == "__main__":
    model = SpectralDecoder()
    x = torch.randn(2, 3, 1024, 1024)
    out = model(x)
    print(f"output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.2f}M")
