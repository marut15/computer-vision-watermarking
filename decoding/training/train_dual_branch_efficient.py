"""Efficient trainer for DualBranchDecoder variants.

What this changes vs. the generic ``train.py``:

* **bf16 AMP** via ``torch.autocast`` (no GradScaler needed on Blackwell /
  Ampere). Roughly halves wall time and frees VRAM, which lets us push the
  batch size up.
* **Cosine LR with linear warm-up.** The R-18 dual_branch run had three
  wasted epochs of flat loss before the spatial branch broke symmetry; a
  short warm-up keeps the first epochs from being a no-op while cosine
  decay lets the late epochs anneal cleanly.
* **On-the-fly JPEG augmentation.** The single biggest robustness gap in
  the previous results was JPEG q75 (DualBranch dropped to 0.589). We
  inject a random-quality JPEG round-trip into the train transform with a
  configurable probability so the spectral branch sees DCT-quantised
  inputs at training time. Eval transforms are untouched.
* **Early stopping** on val exact match (configurable patience).
* **Adam + weight decay** instead of plain Adam.

Drop-in compatible with the existing config schema: pass
``--config configs/dual_branch_r50.yaml`` and it will read
``training.{batch_size, num_epochs, learning_rate, weight_decay,
warmup_pct, jpeg_aug_prob, jpeg_aug_qmin, jpeg_aug_qmax,
early_stop_patience}`` plus ``data.{image_size, ...}`` and
``output.{checkpoint, ...}``.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DECODING_ROOT = HERE.parent
sys.path.insert(0, str(DECODING_ROOT))

from src.dataloader import WatermarkDataset
from src.models import get_model
from src.utils import compute_metrics, print_metrics


# ---------------------------- augmentations ---------------------------- #

class RandomJpegQuality:
    """PIL-level JPEG round-trip with a random quality factor.

    Applied before ToTensor / Normalize. With probability ``p`` re-encodes
    the image to JPEG at a quality drawn uniformly from ``[qmin, qmax]``
    and decodes it back; otherwise returns the image unchanged.
    """

    def __init__(self, p: float = 0.5, qmin: int = 60, qmax: int = 95):
        self.p = p
        self.qmin = qmin
        self.qmax = qmax

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------- LR schedule ---------------------------- #

def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                    base_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warm-up to ``base_lr`` over ``warmup_steps``, then cosine
    decay to ``min_lr`` over the remaining steps.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------- main loop ---------------------------- #

def evaluate(model, loader, criterion, device, autocast_dtype):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["bits"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += float(loss.item())
            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true",
                    help="Disable bf16 autocast (debug only)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override config training.num_epochs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else torch.float32
    use_amp = (device.type == "cuda" and not args.no_amp)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"{'=' * 60}")
    print(f"  device:        {device}")
    print(f"  amp:           {'bf16' if use_amp else 'off'}")
    print(f"  arch:          {cfg['model']['architecture']}")

    image_size = int(cfg["data"]["image_size"])
    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    num_epochs = int(args.max_epochs or train_cfg["num_epochs"])
    base_lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    warmup_pct = float(train_cfg.get("warmup_pct", 0.05))
    jpeg_p = float(train_cfg.get("jpeg_aug_prob", 0.0))
    jpeg_qmin = int(train_cfg.get("jpeg_aug_qmin", 60))
    jpeg_qmax = int(train_cfg.get("jpeg_aug_qmax", 95))
    patience = int(train_cfg.get("early_stop_patience", 0))

    print(f"  image_size:    {image_size}")
    print(f"  batch_size:    {batch_size}")
    print(f"  epochs:        {num_epochs}")
    print(f"  lr:            {base_lr}  (warmup_pct={warmup_pct})")
    print(f"  weight_decay:  {weight_decay}")
    print(f"  jpeg_aug:      p={jpeg_p}  q in [{jpeg_qmin}, {jpeg_qmax}]")
    print(f"  early_stop:    patience={patience}\n")

    # Train transform (with JPEG aug); val transform (clean).
    train_pipeline = [transforms.Resize((image_size, image_size))]
    if jpeg_p > 0:
        train_pipeline.append(RandomJpegQuality(p=jpeg_p, qmin=jpeg_qmin, qmax=jpeg_qmax))
    train_pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transform = transforms.Compose(train_pipeline)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cfg_dir = DECODING_ROOT
    metadata_path = os.path.join(cfg_dir, cfg["data"]["metadata_path"])
    images_path = os.path.join(cfg_dir, cfg["data"]["images_path"])
    splits_path = os.path.join(cfg_dir, cfg["data"]["splits_path"])

    train_full = WatermarkDataset(metadata_path=metadata_path, image_dir=images_path,
                                  transform=train_transform)
    val_full = WatermarkDataset(metadata_path=metadata_path, image_dir=images_path,
                                transform=val_transform)
    with open(splits_path) as f:
        splits = json.load(f)
    train_set = Subset(train_full, splits["train"])
    val_set = Subset(val_full, splits["val"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.num_workers > 0), drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=(args.num_workers > 0))

    print(f"  train: {len(train_set)} images  |  val: {len(val_set)} images")

    model = get_model(architecture=cfg["model"]["architecture"], num_outputs=8,
                      pretrained=cfg["model"].get("pretrained", True)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model params: {n_params:.2f}M\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_pct * total_steps))

    ckpt_path = os.path.join(cfg_dir, cfg["output"]["checkpoint"])
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_exact = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(num_epochs):
        epoch_t0 = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0
        for step_in_epoch, batch in enumerate(train_loader):
            global_step = epoch * steps_per_epoch + step_in_epoch
            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            images = batch["image"].to(device, non_blocking=True)
            targets = batch["bits"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, criterion, device, autocast_dtype)
        epoch_dt = time.time() - epoch_t0

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}  ({epoch_dt:.1f}s)  lr={lr:.6f}")
        print(f"{'=' * 60}")
        print(f"\nTrain loss: {train_loss:.4f}")
        print_metrics(val_metrics, prefix="Val ")

        history.append({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_bit_accuracy": val_metrics["mean_bit_accuracy"],
            "val_exact_match_rate": val_metrics["exact_match_rate"],
            "wall_seconds": epoch_dt,
        })

        if val_metrics["exact_match_rate"] > best_exact:
            best_exact = val_metrics["exact_match_rate"]
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "metrics": val_metrics,
                "history": history,
            }, ckpt_path)
            print(f"\n+ saved best (val_exact={best_exact:.4f})")
        else:
            epochs_since_improve += 1
            print(f"\n  no improvement for {epochs_since_improve} epoch(s); best={best_exact:.4f} @ epoch {best_epoch}")
            if patience and epochs_since_improve >= patience:
                print(f"\n[early-stop] patience {patience} exhausted; stopping at epoch {epoch + 1}.")
                break

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best val exact match: {best_exact:.4f} at epoch {best_epoch}")
    print(f"Checkpoint:           {ckpt_path}")
    print(f"{'=' * 60}\n")

    # Persist the per-epoch history alongside the markdown sink.
    history_path = os.path.join(cfg_dir, "results", "training_logs",
                                f"{cfg['experiment']['name']}.history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({"experiment": cfg["experiment"]["name"],
                   "best_val_exact_match": best_exact,
                   "best_epoch": best_epoch,
                   "history": history}, f, indent=2)
    print(f"history -> {history_path}")


if __name__ == "__main__":
    main()
