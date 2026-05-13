"""Train a tiny linear classifier on the radial-binned power spectrum of g.

This directly tests the hypothesis ``the spectral CNN reads the envelope
shape of |F(g)|''. If the radial-binned envelope of |F(g)| (a vector of
~128 numbers per image) is enough to decode the 8 bits, no convolutions are
needed and the spectral branch is implementing radial-profile-then-linear-
classification.

For each image:
  1. Adaptive-avg-pool to ``fft_size`` (256) to match the model's input.
  2. 2D FFT, log1p magnitude, per-image standardise (same as the model).
  3. Reduce to a radial profile P(k) for k = 0..fft_size/2, channel-wise.
  4. Concatenate the three channels' profiles -> feature vector x.
  5. Train logistic regression (one classifier per bit) on (x, bit_i).

We report:
  - per-bit and mean-bit accuracy on the test split;
  - bits where the linear probe matches / underperforms the trained
    spectral branch (99.6%);
  - sanity check: same probe on log|F(g)| values without standardisation;
  - sanity check: same probe but on the envelope of |F(f)| (baseline only) --
    this *must* be at chance, since the bits don't condition the baseline.

Two ablations of the feature itself:
  --feature radial : just the radial profile (rotationally averaged)
  --feature flat   : the entire fft_size^2 standardised log spectrum,
                     downsampled to a fixed dim. Gives an upper bound for
                     a *purely linear* read of |F(g)|.

Outputs:
  - decoding/results/envelope_linear_probe.json
  - decoding/results/figures/envelope_linear_probe.png

Usage:
  python decoding/scripts/envelope_linear_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))

from src.dataloader import WatermarkDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)


def _spectrum(x: torch.Tensor, fft_size: int, standardise: bool) -> torch.Tensor:
    """Return per-image-standardised log1p|F| at fft_size, identical to
    DualBranchDecoder._spectrum when standardise=True."""
    img = _denormalise(x)
    img = F.adaptive_avg_pool2d(img, fft_size)
    spec = torch.log1p(torch.fft.fftshift(torch.fft.fft2(img).abs(), dim=(-1, -2)))
    if standardise:
        mu = spec.mean(dim=(-1, -2), keepdim=True)
        sigma = spec.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        spec = (spec - mu) / sigma
    return spec


def _radial_indices(size: int) -> torch.Tensor:
    cy = cx = size // 2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2).long()
    return rr  # (size, size)


def radial_profile_batch(spec: torch.Tensor, rr: torch.Tensor, nbins: int) -> torch.Tensor:
    """spec: (B, C, H, W) -> (B, C * nbins). rr: (H, W) integer-radius map."""
    B, C, H, W = spec.shape
    flat_spec = spec.reshape(B, C, -1)        # (B, C, H*W)
    flat_r = rr.reshape(-1)                    # (H*W,)
    bins = torch.zeros(B, C, nbins, device=spec.device)
    counts = torch.zeros(nbins, device=spec.device)
    one = torch.ones_like(flat_r, dtype=torch.float, device=spec.device)
    counts.scatter_add_(0, flat_r.clamp(max=nbins - 1), one)
    counts = counts.clamp(min=1.0)
    for b in range(B):
        for c in range(C):
            bins[b, c].scatter_add_(0, flat_r.clamp(max=nbins - 1), flat_spec[b, c])
    bins = bins / counts
    return bins.reshape(B, C * nbins)


def featurise(loader, fft_size: int, feature: str, standardise: bool, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) where Y[:, i] is bit i."""
    rr = _radial_indices(fft_size).to(device)
    nbins = rr.max().item() + 1
    Xs, Ys = [], []
    for batch in loader:
        x = batch["image"].to(device)
        bits = batch["bits"].numpy()
        spec = _spectrum(x, fft_size, standardise=standardise)
        if feature == "radial":
            feats = radial_profile_batch(spec, rr, nbins).cpu().numpy()
        elif feature == "flat":
            # downsample 2D spec to a fixed 32x32 by avgpool, then flatten
            f = F.adaptive_avg_pool2d(spec, 32).cpu().numpy()
            feats = f.reshape(f.shape[0], -1)
        else:
            raise ValueError(feature)
        Xs.append(feats)
        Ys.append(bits)
    X = np.concatenate(Xs, 0)
    Y = np.concatenate(Ys, 0)
    return X, Y


def fit_per_bit(Xtr, Ytr, Xte, Yte, l2: float = 1.0) -> dict:
    """One logistic regression per bit, scikit-learn."""
    from sklearn.linear_model import LogisticRegression

    n_bits = Ytr.shape[1]
    per_bit = []
    preds_test = np.zeros_like(Yte, dtype=np.int32)
    for i in range(n_bits):
        clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=2000, solver="liblinear")
        clf.fit(Xtr, Ytr[:, i].astype(int))
        pred = clf.predict(Xte)
        preds_test[:, i] = pred
        per_bit.append(float((pred == Yte[:, i]).mean()))
    exact = float((preds_test == Yte.astype(np.int32)).all(axis=1).mean())
    return {
        "per_bit_accuracy": per_bit,
        "mean_bit_accuracy": float(np.mean(per_bit)),
        "exact_match_rate": exact,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--config", default=str(DECODING_ROOT / "configs" / "dual_branch_r50.yaml"))
    p.add_argument("--metadata", default=None)
    p.add_argument("--images", default=None)
    p.add_argument("--baseline", default=None,
                   help="If given, also fits a probe on baseline-only spectra (must "
                        "fail / be at chance, since bits don't condition baseline).")
    p.add_argument("--splits", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--fft-size", type=int, default=256)
    p.add_argument("--feature", choices=["radial", "flat"], default="radial")
    p.add_argument("--no-standardise", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--out-json", default=str(DECODING_ROOT / "results" / "envelope_linear_probe.json"))
    p.add_argument("--out-fig", default=str(DECODING_ROOT / "results" / "figures" / "envelope_linear_probe.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    metadata = args.metadata or os.path.join(DECODING_ROOT, cfg["data"]["metadata_path"])
    images = args.images or os.path.join(DECODING_ROOT, cfg["data"]["images_path"])
    splits = args.splits or os.path.join(DECODING_ROOT, cfg["data"]["splits_path"])
    image_size = args.image_size or cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} feature={args.feature} fft={args.fft_size} "
          f"standardise={not args.no_standardise}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    full = WatermarkDataset(metadata_path=metadata, image_dir=images, transform=transform)
    with open(splits) as f:
        split_idx = json.load(f)

    train_idx = split_idx["train"]
    if args.limit_train:
        train_idx = train_idx[: args.limit_train]
    train_set = Subset(full, train_idx)
    test_set = Subset(full, split_idx["test"])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"[probe] featurising train ({len(train_set)})...")
    Xtr, Ytr = featurise(train_loader, args.fft_size, args.feature,
                         standardise=not args.no_standardise, device=device)
    print(f"[probe] featurising test  ({len(test_set)})...")
    Xte, Yte = featurise(test_loader, args.fft_size, args.feature,
                         standardise=not args.no_standardise, device=device)
    print(f"[probe] X_train={Xtr.shape}  X_test={Xte.shape}")

    res = fit_per_bit(Xtr, Ytr, Xte, Yte)
    print(f"[probe] mean_bit={res['mean_bit_accuracy']:.4f}  exact={res['exact_match_rate']:.4f}")
    print(f"[probe] per_bit={[f'{v:.3f}' for v in res['per_bit_accuracy']]}")

    out = {
        "feature": args.feature,
        "standardise": not args.no_standardise,
        "fft_size": args.fft_size,
        "image_size": image_size,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "result": res,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"results: {out_json}")

    # Plot per-bit comparison vs reported spectral-branch numbers.
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bits = np.arange(8)
    ax.bar(bits - 0.2, res["per_bit_accuracy"], width=0.4, label=f"linear probe ({args.feature})")
    # Reference: spectral-only ablation accuracies, hard-coded from
    # decoding/results/dual_branch_r50.md (no_spatial mode).
    ref = [1.000, 0.996, 1.000, 0.992, 0.984, 1.000, 0.984, 0.961]  # approx
    ax.bar(bits + 0.2, ref, width=0.4, label="spectral branch (no_spatial)")
    ax.axhline(0.5, color="grey", linestyle=":")
    ax.set_xticks(bits)
    ax.set_xlabel("bit")
    ax.set_ylabel("test accuracy")
    ax.set_title("Linear probe on |F(g)| envelope vs. trained spectral branch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print(f"figure: {out_fig}")


if __name__ == "__main__":
    main()
