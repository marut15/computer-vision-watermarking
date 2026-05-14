"""GlobalStatsDecoder — decode watermark bits from explicit global image statistics.

The LoRA style-slider watermarks modulate *global* image statistics: bit 3
(bright/dark) is a luminance shift, bit 4 (contrast) is a per-channel std
shift, bit 5 (saturation) is a chroma-vs-luma std ratio, bit 0 (warm/cool)
is a red-vs-blue mean shift. A standard CNN with global average pooling at
the end implicitly recovers these, but its inductive bias favours local
texture (early layers see only 3×3 / 7×7 patches; only the final pool
collapses to a global view). Grad-CAM confirms the trained ResNet attends to
narrow image regions.

This decoder makes the global-statistic hypothesis explicit. It computes a
fixed-size vector of hand-crafted statistics — per-channel moments and
percentiles, channel-difference means, gradient / laplacian energy, radial
FFT-band energy — then runs only an MLP on top. No spatial conv layers at
all. By construction every input pixel contributes to every feature, so
there is no local-patch bias to overcome.

If this trivial model competes with ResNet-50 on the bright/dark bit, that
is itself the result: the watermark really is a global statistic shift, and
the limiting factor for a CNN is that it has to re-derive these statistics
from a pile of local convolutions instead of being handed them directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GlobalStatsDecoder(nn.Module):
    NUM_BITS = 8

    def __init__(
        self,
        num_outputs: int = 8,
        num_fft_bands: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_fft_bands = num_fft_bands

        # Buffers move with .to(device); no need to recreate tensors every forward.
        self.register_buffer("imnet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imnet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # 3×3 Laplacian kernel, replicated per channel for grouped conv.
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        # Luminance weights (BT.601).
        self.register_buffer("luma_w", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))
        # Percentile cut-points used for per-channel quantiles.
        self.register_buffer("pct_q", torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))

        # Feature counts:
        # per-channel mean (3) + std (3) + 5 percentiles × 3 (15)
        #   + gradient energy × 3 + laplacian energy × 3              = 27
        # channel-diff means (R-G, R-B, G-B)                          =  3
        # luminance mean / std + chroma std                            =  3
        # radial FFT band energies × 3 channels                        = num_fft_bands * 3
        self.num_stats = 27 + 3 + 3 + num_fft_bands * 3

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.num_stats),
            nn.Linear(self.num_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.imnet_std + self.imnet_mean).clamp(0.0, 1.0)

    def _radial_fft_bands(self, img: torch.Tensor) -> torch.Tensor:
        """Mean log-magnitude of centred 2D FFT in `num_fft_bands` radial bins."""
        B, C, H, W = img.shape
        fft = torch.fft.fft2(img)
        mag = torch.fft.fftshift(fft.abs(), dim=(-1, -2))
        mag = torch.log1p(mag)

        device, dtype = img.device, img.dtype
        yy = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0
        xx = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0
        rr = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        rr = rr / (rr.max() + 1e-8)  # 0..1

        edges = torch.linspace(0.0, 1.0, self.num_fft_bands + 1, device=device, dtype=dtype)
        bands = []
        for i in range(self.num_fft_bands):
            lo, hi = edges[i], edges[i + 1]
            if i == self.num_fft_bands - 1:
                mask = ((rr >= lo) & (rr <= hi)).to(dtype)
            else:
                mask = ((rr >= lo) & (rr < hi)).to(dtype)
            denom = mask.sum().clamp_min(1.0)
            band = (mag * mask).sum(dim=(-1, -2)) / denom  # (B, C)
            bands.append(band)
        return torch.stack(bands, dim=-1).reshape(B, -1)  # (B, C * num_fft_bands)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        img = self._denormalize(x)
        B, C, H, W = img.shape
        flat = img.reshape(B, C, -1)

        chan_mean = flat.mean(dim=-1)
        chan_std = flat.std(dim=-1, unbiased=False)

        # Quantiles per (batch, channel). torch.quantile over the last dim.
        chan_pct = torch.quantile(flat, self.pct_q, dim=-1)  # (5, B, C)
        chan_pct = chan_pct.permute(1, 2, 0).reshape(B, -1)  # (B, C*5)

        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = img[..., 1:, :] - img[..., :-1, :]
        grad_energy = gx.abs().mean(dim=(-1, -2)) + gy.abs().mean(dim=(-1, -2))  # (B, C)

        lap = F.conv2d(img, self.lap_kernel, groups=C, padding=1)
        lap_energy = lap.abs().mean(dim=(-1, -2))  # (B, C)

        chan_diff = torch.stack(
            [
                chan_mean[:, 0] - chan_mean[:, 1],  # warm/cool proxy
                chan_mean[:, 0] - chan_mean[:, 2],
                chan_mean[:, 1] - chan_mean[:, 2],
            ],
            dim=-1,
        )

        lum = (img * self.luma_w).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        lum_mean = lum.mean(dim=(-1, -2)).squeeze(-1)
        lum_std = lum.std(dim=(-1, -2), unbiased=False).squeeze(-1)
        chroma = img - lum  # broadcast: per-channel deviation from luminance
        chroma_std = chroma.std(dim=(-1, -2), unbiased=False).mean(dim=-1)
        lc_stats = torch.stack([lum_mean, lum_std, chroma_std], dim=-1)

        fft_bands = self._radial_fft_bands(img)

        return torch.cat(
            [chan_mean, chan_std, chan_pct, grad_energy, lap_energy, chan_diff, lc_stats, fft_bands],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._compute_stats(x))


if __name__ == "__main__":
    model = GlobalStatsDecoder()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"output shape: {out.shape}")
    print(f"num stats: {model.num_stats}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.3f}M")
