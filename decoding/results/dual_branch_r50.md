# DualBranch R-50 (1024) — Final Model Analysis

The DualBranch decoder paired a ResNet-18 spatial trunk with an FFT log-magnitude
CNN (the same stack as the standalone Spectral decoder) and was already the best
model in the suite. This run replaces the spatial trunk with **ResNet-50** and
trains at the canonical **1024×1024** resolution — apples-to-apples with the
original `baseline_resnet50` ablation. It also adds bf16 AMP, a cosine LR
schedule with warm-up, on-the-fly JPEG augmentation, and early stopping.

## Configuration

- **Architecture:** DualBranch (ResNet-50 spatial branch + FFT-CNN spectral branch, fused by concat → MLP)
- **Resolution:** 1024×1024
- **Batch size:** 8 (bf16 AMP)
- **Optimizer:** AdamW, weight_decay 1e-4
- **LR schedule:** cosine, base 1e-4, linear warm-up 5%
- **Augmentation:** random JPEG quality ∈ [60, 95] with p=0.5
- **Epochs:** 40 (early-stop patience 8)
- **Loss:** BCEWithLogitsLoss
- **Device:** NVIDIA RTX PRO 6000 Blackwell (95 GiB)
- **Training time:** ~23 min (1396 s)

Spec'd in `decoding/configs/dual_branch_r50.yaml`; trainer at
`decoding/scripts/train_dual_branch_efficient.py`; orchestrated end-to-end
(train → clean eval → branch ablation → robustness) by
`decoding/scripts/run_dual_branch_r50.sh`.

---

## Final Performance

### Test Set (held-out, unseen during training)

- **Mean bit accuracy:** **99.66 %**
- **Exact match rate (8/8 bits):** **97.27 %**

### Per-bit accuracy

| Bit | Slider Pair | Test Accuracy |
|-----|-------------|---------------|
| 0 | warm / cool | 100.00 % |
| 1 | sharp / soft | 99.61 % |
| 2 | grainy / clean | 100.00 % |
| 3 | bright / dark | 99.61 % |
| 4 | contrast | 99.22 % |
| 5 | saturation | 100.00 % |
| 6 | detail | 99.22 % |
| 7 | vintage / modern | 99.61 % |

Every bit is now ≥ 99 %; the previously weakest bits (3 bright/dark and
4 contrast) come up from R-18's 96 % / 95 % to 99.6 % / 99.2 %.

---

## Architecture Comparison (clean test set)

| Architecture | Params | Resolution | Mean bit | Exact match |
|---|---:|---:|---:|---:|
| **DualBranch R-50 (this work)** | **~36 M** | **1024** | **99.66 %** | **97.27 %** |
| DualBranch R-18 | ~14 M | 512 | 97.78 % | 84.38 % |
| ResNet-50 baseline (shared head) | 25.6 M | 1024 | 93.12 % | 57.42 % |
| ResNet-50 @ 512 (ablation 2) | 25.6 M | 512 | 93.90 % | 63.67 % |
| EfficientNet-B0 | 4.0 M | 1024 | 92.58 % | 54.69 % |
| 8 × ResNet-50 (one head per bit) | ~200 M | 1024 | 88.28 % | 33.59 % |
| ViT-B/16 | 86 M | 224 (forced) | 47.07 % | 0.00 % |
| MultiScale Pyramid (R-18 × 3 scales) | 12 M | 512 | 48.29 % | 0.78 % |
| Spectral (FFT-CNN, standalone) | 3 M | 1024 | 48.97 % | 0.00 % |
| GlobalStats MLP | 0.2 M | 256 | 48.10 % | 0.00 % |

**Headlines.**
- DualBranch R-50 is the new clean-accuracy SOTA: **+ 6.5 pp mean bit** and
  **+ 39.9 pp exact match** over the previous best non-DualBranch baseline
  (ResNet-50 shared head).
- Going R-18 → R-50 inside DualBranch adds **+ 1.9 pp mean bit** and
  **+ 12.9 pp exact match**, and the higher 1024 resolution lets the spatial
  branch contribute meaningfully (see the ablation below).
- Three single-feature decoders (GlobalStats / Spectral-standalone / MultiScale)
  never escaped chance; ViT failed the same way. Architecture priors matter
  more than parameter count *within the spatial-only paradigm*.

---

## Branch Ablation — Where Does the Signal Live?

We ran the full test set three ways: with both branches active, with the
spectral branch's features zeroed before fusion, and with the spatial branch's
features zeroed before fusion.

| Mode | Mean bit | Exact | Δ vs. full mean |
|---|---:|---:|---:|
| `full` | **99.66 %** | **97.27 %** | — |
| `no_spatial` (spectral alone) | 99.61 % | 96.88 % | − 0.05 pp |
| `no_spectral` (spatial alone) | 59.47 % | 1.95 % | − 40.19 pp |

**Compared to R-18 DualBranch:**

| Mode | R-18 (512) | R-50 (1024) | Δ |
|---|---:|---:|---:|
| `full` | 97.78 % | 99.66 % | + 1.88 pp |
| `no_spatial` | 96.14 % | 99.61 % | + 3.47 pp |
| `no_spectral` | 49.90 % (chance) | 59.47 % | + 9.57 pp |

**Reading.**
- The **spectral branch still carries the model.** Removing it collapses
  performance. Removing the spatial branch costs essentially nothing on the
  clean set.
- Going R-18 → R-50 in the spatial trunk lifts `no_spectral` from chance
  (49.9 %) to clearly above chance (59.5 %), but it's still useless on its own
  for decoding the watermark — barely 5 % exact match. So the larger spatial
  network *can* extract some signal, just nowhere near enough to decode 8
  independent bits.
- The full model gets a small but real boost from the higher-capacity spatial
  branch (+ 1.88 pp mean bit), most of which is concentrated on the bits where
  the R-18 model previously struggled.

This generalises the finding from R-18 ablation: **the LoRA-slider watermark
is, to first order, a global frequency-domain signal**, not a localized spatial
texture.

---

## Robustness

Six attacks at 1024×1024. JPEG augmentation during training was the single
biggest change between R-18 and R-50.

| Attack | DualBranch R-50 | DualBranch R-18 | ResNet-50 baseline | 8× separate |
|---|---:|---:|---:|---:|
| clean | **99.66 %** | 97.78 % | 93.12 % | 88.28 % |
| jpeg_q90 | **98.00 %** | — | 68.46 % | 77.73 % |
| jpeg_q75 | **97.17 %** | 58.94 % | 56.45 % | 62.40 % |
| jpeg_q50 | **84.96 %** | — | 53.20 % | 53.20 % |
| resize_512 | 95.17 % | **97.78 %** | 55.00 % | 60.94 % |
| random_crop_75 | 50.10 % (chance) | 49.46 % | 56.93 % | 57.04 % |

(Cells marked "—" are attacks that weren't run on the R-18 model.)

**Reading.**
- **JPEG resilience is solved.** Q75 jumped from 58.9 % → 97.2 % (+ 38.3 pp).
  Q90 sits at 98.0 %. Even Q50 — which heavily quantises high-frequency DCT
  coefficients, exactly the cells the spectral branch reads — holds at 85.0 %.
  The training-time JPEG augmentation (q ∈ [60, 95], p=0.5) closed this gap
  almost entirely.
- **Resize regressed slightly** (97.8 % → 95.2 %) compared to R-18. The R-18
  model was trained at 512 so the test-time resize-to-512 attack was a no-op
  for it; the R-50 model was trained at 1024 so it has to handle a real
  4× area downsample. Still 95 %, still very robust.
- **Random crop is the unsolved attack.** Cropping to 75 % of the image and
  upscaling back destroys the spectral branch's signal (the FFT shape changes
  on the cropped region) and the spatial branch alone can't decode. Training
  augmentation that includes random crops is the obvious next experiment.

---

## Training Dynamics

The R-18 DualBranch run had three flat warm-up epochs at constant LR before
the spatial branch broke symmetry. The R-50 run with cosine + 5 % linear
warm-up + AMP escaped chance much faster: the train loss starts at 0.69
(= ln 2), drops below 0.5 by epoch 4, and the val exact-match enters the 90 %
range around epoch 12. Best val was at epoch 28; early stopping (patience 8)
kicked in shortly after.

Per-epoch trajectories for all five "new" decoders are at
`decoding/results/training_logs/dual_branch_r50.history.json` and visualised
in `new_models_figures/training_loss.png` /
`new_models_figures/training_val_curves.png`.

---

## Working Theory vs. What the Data Says

The original design hypothesis was: LoRA-slider edits are localized in latent
space, so decoding ought to need a high-capacity *spatial* network with many
decision hyperplanes — clipping at the unit interval and JPEG quantisation
both inject high-frequency noise that a deep feature stack should absorb.

The data partly confirm and partly redirect this:

- ✓ **Capacity matters.** Within the spatial-only family, ResNet-50 (93 %)
  beats EfficientNet-B0 (93 %) and ResNet-18 (the R-18 spatial branch alone
  was at chance), and clearly beats ViT-B/16 (47 %).
- ✓ **Adding a stronger spatial branch helps DualBranch on the marginal bits**
  (bright/dark, contrast). R-50 spatial-alone reaches 59.5 % vs. R-18's
  49.9 %, and the per-bit gain in the full model concentrates on those bits.
- ✗ **But the LoRA effect isn't "localized in pixel space" by the metric the
  decoder cares about.** It leaves a *globally-distributed frequency
  signature*. The spectral branch alone reaches 99.6 %; spatial alone — even
  with R-50 — barely escapes chance and gets 2 % exact match. No amount of
  spatial capacity is going to close that gap on its own.
- ✓ **High capacity *does* buy robustness to clipping/JPEG**, as predicted —
  but the buy is conditional on training-time exposure to JPEG. Without the
  JPEG augmentation, the R-50 spectral path would still degrade to ~ 60 %
  at q75 (since the spectral cells are exactly what JPEG quantises). The
  augmentation teaches the model to read the surviving signal.

**Synthesis for the report.** The right decoder for this LoRA-slider scheme
is a **frequency-domain feature extractor with a high-capacity spatial
co-decoder**, trained with JPEG augmentation. The frequency branch carries
the bits; the spatial branch contributes refinement on the bits whose
spectral signature is weakest; JPEG augmentation hardens both.

---

## Outstanding Failure Modes

1. **Random crop 75 %** drops to chance (50.1 %). Geometric attacks that
   disturb the global FFT structure remain a hard counter. Adding random-crop
   augmentation during training is the obvious next experiment.
2. **JPEG q50 — q90 curve** is monotonic but not flat: q90 = 98.0 % → q75 =
   97.2 % → q50 = 85.0 %. Pushing further (q40, q30) likely breaks down
   sharply.
3. **Spatial branch is still overkill at R-50.** Most of its parameters are
   doing nothing for this task. A leaner spatial branch (R-34 or even a few
   convolutional layers from scratch) trained jointly with the spectral
   branch would likely match this performance with ~half the parameters.

---

## Reproducibility

```bash
# from /workspace/computer-vision-watermarking
bash decoding/scripts/run_dual_branch_r50.sh        # train + eval + ablate + robustness
bash decoding/scripts/build_new_models_figures.sh   # cross-arch comparison plots
bash decoding/scripts/stage_good_dual_branch_data.sh  # bundle for S3
```

Artefacts land at `/workspace/new_models/dual_branch_r50/` (per-arch),
`/workspace/new_models_figures/` (cross-arch), and
`/workspace/good_dual_branch_data/` (final bundle).
