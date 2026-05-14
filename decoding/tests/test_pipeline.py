#!/usr/bin/env python3
"""Pipeline dry-run test: decoder CLI, evaluate, robustness, encoder.

Tests (all run against real data + GPU, but small batches for speed):

  1. Model imports & instantiation (all architectures)
  2. Decoder forward pass on real data (GPU, 1 batch, no checkpoint)
  3. Decoder forward pass with real dual_branch_r50 checkpoint (GPU, 1 batch)
  4. evaluate.py --smoke (2 batches, real checkpoint, real data, GPU)
  5. robustness_eval.py --smoke (synthetic fixture, fast)
  6. Encoder dry run (no SDXL needed: verify configs, LoRA files, import gate)

Run from repo root:
    cd /workspace/repo/computer-vision-watermarking
    python decoding/tests/test_pipeline.py

Exit 0 = all passed, 1 = one or more failed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = REPO / "evaluation/scripts"
sys.path.insert(0, str(REPO))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures: list[str] = []


def ok(label: str, detail: str = ""):
    print(f"  {PASS}  {label}" + (f"\n        {detail}" if detail else ""))


def fail(label: str, detail: str = ""):
    print(f"  {FAIL}  {label}" + (f"\n        {detail}" if detail else ""))
    failures.append(label)


def check(label: str, condition: bool, detail: str = ""):
    (ok if condition else fail)(label, detail)


def skip(label: str, reason: str = ""):
    print(f"  {SKIP}  {label}" + (f" ({reason})" if reason else ""))


def section(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


def run(cmd: list[str], cwd=None, timeout=120, extra_env: dict | None = None) -> tuple[int, str, float]:
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PROJECT_DATA_ROOT", str(DATA))
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(cwd or REPO / "decoding"),
        timeout=timeout, env=env,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    return result.returncode, output, elapsed


# ── shared setup ─────────────────────────────────────────────────────────────
from project_paths import Paths
p = Paths()
DATA = p.data_root

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
print(f"  data root: {DATA}")


# ── 1. Model imports & instantiation ─────────────────────────────────────────
section("1. Model imports & instantiation (all architectures)")
try:
    from decoding.models import get_model
    ok("src.models imports")
except Exception as e:
    fail("src.models imports", str(e))

arch_cases = [
    ("resnet50", {}),
    ("dual_branch", {}),
    ("dual_branch_r50", {}),
    ("spectral", {}),
    ("global_stats", {}),
    ("multiscale_pyramid", {}),
]
for arch, kw in arch_cases:
    try:
        m = get_model(arch, pretrained=False, **kw)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        ok(f"get_model('{arch}')  [{n:.1f}M params]")
    except Exception as e:
        fail(f"get_model('{arch}')", str(e))


# ── 2. Decoder forward pass on real data (no checkpoint) ─────────────────────
section("2. Decoder forward pass on real data — no checkpoint (GPU)")
try:
    from torchvision import transforms
    from decoding.data.dataset import WatermarkDataset
    from torch.utils.data import DataLoader, Subset

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = WatermarkDataset(str(p.metadata), str(p.images_dir), transform=tf)
    with open(REPO / "decoding/data/splits.json") as f:
        splits = json.load(f)
    loader = DataLoader(Subset(ds, splits["test"][:8]), batch_size=8, num_workers=2)
    batch = next(iter(loader))

    model = get_model("dual_branch_r50", pretrained=False).to(DEVICE).eval()
    with torch.no_grad():
        logits = model(batch["image"].to(DEVICE))
    check("forward pass shape == (8, 8)", tuple(logits.shape) == (8, 8))
    ok(f"logits range [{logits.min():.2f}, {logits.max():.2f}] (random weights — values don't matter)")
except Exception as e:
    fail("forward pass on real data", str(e))


# ── 3. Load dual_branch_r50 checkpoint & run inference ───────────────────────
section("3. Load dual_branch_r50 checkpoint + real inference (GPU, 8 images)")
ckpt_path = p.checkpoint("dual_branch_r50")
if not ckpt_path.exists():
    skip("checkpoint inference", f"not found: {ckpt_path}")
else:
    try:
        t0 = time.time()
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ok(f"checkpoint loaded in {time.time()-t0:.1f}s  (epoch {ckpt.get('epoch','?')})")

        model = get_model("dual_branch_r50", pretrained=False).to(DEVICE).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        ok("model_state_dict loaded cleanly")

        # Run on 8 real test images at 1024px (same as training res)
        tf_1024 = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds_1024 = WatermarkDataset(str(p.metadata), str(p.images_dir), transform=tf_1024)
        loader_1024 = DataLoader(Subset(ds_1024, splits["test"][:8]), batch_size=8, num_workers=2)
        batch = next(iter(loader_1024))

        with torch.no_grad():
            logits = model(batch["image"].to(DEVICE))
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()

        targets = batch["bits"]
        bit_acc = (preds == targets).float().mean().item()
        exact = ((preds == targets).all(dim=1)).float().mean().item()
        ok(f"inference complete  bit_acc={bit_acc:.3f}  exact_match={exact:.3f}  (8-sample subset)")
        check("bit accuracy > 95% on real data", bit_acc > 0.95,
              f"got {bit_acc:.3f} — model may not be dual_branch_r50 weights")
    except Exception as e:
        fail("checkpoint inference", str(e))
        import traceback; traceback.print_exc()


# ── 4. evaluate.py --smoke (2 batches, real data + real checkpoint) ──────────
section("4. evaluate.py --smoke  (2 batches, real checkpoint, GPU)")
cfg_path = REPO / "decoding/configs/dual_branch_r50.yaml"
rc, out, elapsed = run(
    [sys.executable, str(EVAL_SCRIPTS / "evaluate.py"),
     "--config", str(cfg_path), "--smoke"],
    cwd=REPO / "decoding",
    timeout=180,
)
print(f"  exit={rc}  elapsed={elapsed:.1f}s")
# Show last 15 lines
for line in out.strip().splitlines()[-15:]:
    print(f"    {line}")
check("evaluate.py --smoke exits 0", rc == 0, out[-300:] if rc != 0 else "")


# ── 5. robustness_eval.py --smoke (synthetic fixture, fast) ──────────────────
section("5. robustness_eval.py --smoke  (synthetic fixture, no GPU needed)")
rc, out, elapsed = run(
    [sys.executable, str(EVAL_SCRIPTS / "robustness_eval.py"), "--smoke", "--model", "resnet"],
    cwd=REPO / "decoding",
    timeout=300,
)
print(f"  exit={rc}  elapsed={elapsed:.1f}s")
for line in out.strip().splitlines()[-12:]:
    print(f"    {line}")
check("robustness_eval.py --smoke exits 0", rc == 0, out[-300:] if rc != 0 else "")


# ── 6. Encoder dry run (no SDXL) ─────────────────────────────────────────────
section("6. Encoder dry run  (no SDXL generation — config + LoRA file checks)")

# 6a. Encoding configs
enc_configs = list((REPO / "encoding/configs").glob("*.yaml"))
check(f"encoding configs present ({len(enc_configs)} found)", len(enc_configs) == 8)

if enc_configs:
    try:
        import yaml as _yaml
        for cfg in sorted(enc_configs):
            _yaml.safe_load(cfg.read_text())
        ok("all 8 encoding configs parse as valid YAML")
    except Exception as e:
        fail("encoding configs parse", str(e))

# 6b. Encoding prompts
enc_prompts = list((REPO / "encoding/prompts").glob("*.yaml"))
check(f"encoding prompts present ({len(enc_prompts)} found)", len(enc_prompts) == 8)

# 6c. LoRA safetensors in data root
lora_enc = list((DATA / "encoding/models").rglob("*.safetensors"))
check(f"encoding LoRA safetensors in data ({len(lora_enc)} found, expect 16)", len(lora_enc) == 16)
lora_wm = list((DATA / "watermark_encoding/models").rglob("*.safetensors"))
check(f"watermark_encoding LoRA safetensors in data ({len(lora_wm)} found, expect 8)", len(lora_wm) == 8)

# 6d. generate_dataset.py syntax check (no import of diffusers)
gen_script = REPO / "encoding/scripts/generate_dataset.py"
rc_syn, out_syn, _ = run(
    [sys.executable, "-m", "py_compile", str(gen_script)],
    cwd=REPO, timeout=10,
)
check("generate_dataset.py syntax is valid", rc_syn == 0, out_syn)

# 6e. Report what a real run would do
with open(gen_script) as f:
    src = f.read()
n_ids = src.count("id_int") and "256 IDs × 10 prompts = 2560 images"
try:
    import yaml as _yaml
    n_prompts = len(_yaml.safe_load((REPO / "encoding/prompts/prompts-watermark-s1.yaml").read_text()))
except Exception:
    n_prompts = "?"
print(f"\n  [dry-run summary] a real encoder run would:")
print(f"    • load SDXL + 8 LoRA sliders from {DATA / 'watermark_encoding/models'}")
print(f"    • generate images → {DATA / 'watermark_encoding/data/images'}")
print(f"    • requires: diffusers, transformers, accelerate, CUDA ~40GB VRAM")
try:
    import diffusers
    print(f"    • diffusers {diffusers.__version__} is installed — ready to run")
except ImportError:
    print(f"    • diffusers NOT installed  (pip install diffusers transformers accelerate)")


# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*62}")
if failures:
    print(f"  {FAIL}  {len(failures)} check(s) failed:")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    print(f"  {PASS}  All pipeline checks passed.")
print(f"{'═'*62}\n")
