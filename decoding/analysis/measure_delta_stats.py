from pathlib import Path
import json
import numpy as np
from PIL import Image

REPO = Path("/workspace/computer-vision-watermarking")
META = json.loads((REPO / "watermark_encoding/data/metadata.json").read_text())
IMG = REPO / "watermark_encoding/data/images"
BAS = REPO / "watermark_encoding/data/baseline"

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

# Δ for every watermarked image vs its prompt's baseline
THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
agg = {t: [] for t in THRESHOLDS}
linfs = []

for entry in META[:200]:                                # subsample for speed
    pidx = entry.get("prompt_idx") or int(entry["file"].split("_p")[1][:2])
    f = load(BAS / f"baseline_p{pidx:02d}.png")
    g = load(IMG / entry["file"])
    if f.shape != g.shape:
        continue
    delta = np.abs(g - f).max(axis=-1)                  # collapse RGB -> max abs
    linfs.append(delta.max())
    for t in THRESHOLDS:
        agg[t].append((delta >= t).mean() * 100)        # % pixels above threshold

print(f"n images:                  {len(linfs)}")
print(f"mean ‖Δ‖∞ (max abs):       {np.mean(linfs):.4f}")
print(f"median ‖Δ‖∞:               {np.median(linfs):.4f}")
print()
print("fraction of pixels with |Δ| ≥ threshold (mean across images, %):")
for t in THRESHOLDS:
    print(f"  ≥ {t*100:>5.1f}% of dynamic range : {np.mean(agg[t]):>6.2f}% of pixels")