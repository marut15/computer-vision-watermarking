# Architecture Comparison: ResNet-50 vs EfficientNet-B0

## Summary

| Model | Test Accuracy | Exact Match | Parameters | Winner |
|-------|---------------|-------------|------------|--------|
| **ResNet-50** | **94.04%** | **60.94%** | 25.6M | ✅ |
| EfficientNet-B0 | 92.58% | 54.69% | 4.0M | ❌ |

**Verdict:** ResNet-50 is superior. EfficientNet-B0 underfits the subtle watermark signals.

## Decision

**Use ResNet-50 as the baseline** for Person B's advanced modeling work.

## Week 2 Status

- ✅ Experiment 1 (Architecture): Complete - ResNet-50 wins
- ⏭️ Experiment 2 (Resolution): Skipped - not enough value vs time
- ⏭️ Experiment 3 (Augmentation): Skipped - likely to hurt performance

**Handoff to Person B:** Baseline model (ResNet-50, 94.04%, 60.94%) is ready.
