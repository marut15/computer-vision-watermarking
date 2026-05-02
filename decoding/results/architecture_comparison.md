# Architecture comparison (held-out test set, 1024 × 1024 native)

256 test samples drawn via `decoding/data/splits.json`. All three architectures evaluated against the same set of test images at the dataset's native resolution; ViT-B/16 is downsampled to its required input size inside the eval loop just before its forward pass.

| Architecture | Bit 0 | Bit 1 | Bit 2 | Bit 3 | Bit 4 | Bit 5 | Bit 6 | Bit 7 | Mean | Exact |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 (shared backbone)        | 0.9023 | 0.9453 | 0.9570 | 0.8281 | 0.9219 | 0.9297 | 0.9961 | 0.9648 | **0.9307** | **0.5703** |
| 8 × ResNet-50 (separate)           | 0.9062 | 0.9219 | 0.9609 | 0.5820 | 0.8203 | 0.9297 | 0.9883 | 0.9531 | 0.8828 | 0.3359 |
| ViT-B/16                           | 0.4258 | 0.5039 | 0.4492 | 0.4883 | 0.4961 | 0.4570 | 0.4727 | 0.4766 | 0.4712 | 0.0000 |

Slider semantics — bit 0: warm/cool · bit 1: sharp/soft · bit 2: grainy/clean · bit 3: bright/dark · bit 4: contrast · bit 5: saturation · bit 6: detail · bit 7: vintage/modern.

## Checkpoints used

- **ResNet-50 (shared backbone)** — `decoding/checkpoints/baseline_resnet50.pth` (loaded)
- **8 × ResNet-50 (separate)** — `decoding/checkpoints/separate/bit_{0..7}_best.pth` (loaded)
- **ViT-B/16** — `decoding/checkpoints/vit_best.pth` (loaded; trained but never escaped chance-level loss — see [decoder_performance.md §4.2](decoder_performance.md))

Reproduce: `python decoding/scripts/compare_architectures.py --batch-size 4`. Full discussion of these numbers, per-bit val curves, robustness, and failure-mode analysis in [decoder_performance.md](decoder_performance.md).
