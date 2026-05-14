#!/usr/bin/env python3
"""Dry-run layout test for the repo/data split migration.

Checks:
  1. project_paths.py resolves all expected paths correctly
  2. Every expected file in /workspace/data exists with non-zero size
  3. No .pth / .safetensors remain in the repo working tree
  4. All decoder YAML configs load and their PROJECT_DATA_ROOT tokens expand
  5. metadata.json is valid JSON with the expected structure
  6. dataloader.py instantiates a WatermarkDataset against real data (no GPU)
  7. Checkpoint files open as valid PyTorch archives (no full load, just header check)

Run from repo root:
    cd /workspace/repo/computer-vision-watermarking
    python decoding/tests/test_layout.py

Or with a custom data root:
    PROJECT_DATA_ROOT=/mnt/data/cvm python decoding/tests/test_layout.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

failures = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"\n        {detail}" if detail else ""))
        failures.append(label)


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. project_paths.py ───────────────────────────────────────────────────────
section("1. project_paths.py")
try:
    from project_paths import Paths
    p = Paths()
    check("Paths() instantiates", True)
    check("data_root is an absolute path", p.data_root.is_absolute())
    check("repo_root points at repo", (p.repo_root / "project_paths.py").exists())
    print(f"       data_root  = {p.data_root}")
    print(f"       repo_root  = {p.repo_root}")

    # env-var override
    os.environ["PROJECT_DATA_ROOT"] = "/tmp/fake_data_root"
    p2 = Paths()
    check("PROJECT_DATA_ROOT env var is honoured", str(p2.data_root) == "/tmp/fake_data_root")
    del os.environ["PROJECT_DATA_ROOT"]

    # explicit override
    p3 = Paths(data_root="/tmp/explicit")
    check("explicit data_root= kwarg overrides env", str(p3.data_root) == "/tmp/explicit")

except Exception as exc:
    check("project_paths.py imports without error", False, str(exc))
    p = None


# ── 2. expected files in /workspace/data ─────────────────────────────────────
section("2. /workspace/data file presence")
if p is None:
    print(f"  {SKIP}  (project_paths failed, skipping data checks)")
else:
    DATA = p.data_root

    expected_files = [
        DATA / "watermark_encoding/data/metadata.json",
        DATA / "decoding/checkpoints/dual_branch_r50.pth",
        DATA / "decoding/checkpoints/baseline_resnet50.pth",
        DATA / "decoding/checkpoints/dual_branch.pth",
        DATA / "decoding/checkpoints/spectral.pth",
        DATA / "decoding/checkpoints/vit_best.pth",
        DATA / "decoding/model_bundles/dual_branch_r50/dual_branch_r50.pth",
        DATA / "decoding/model_bundles/dual_branch_r50/clean_metrics.json",
        DATA / "decoding/model_bundles/dual_branch_r50/robustness.json",
        DATA / "decoding/model_bundles/dual_branch_r50/training_history.json",
        DATA / "manifests/checkpoint_manifest.json",
        DATA / "manifests/data_manifest.md",
    ]
    for f in expected_files:
        check(f"exists: data/{f.relative_to(DATA)}", f.exists() and f.stat().st_size > 0)

    img_count = len(list((DATA / "watermark_encoding/data/images").glob("*.png")))
    check(f"watermark images: {img_count} PNGs (expect 2560)", img_count == 2560)
    baseline_count = len(list((DATA / "watermark_encoding/data/baseline").glob("*.png")))
    check(f"baseline images: {baseline_count} PNGs (expect 10)", baseline_count == 10)

    enc_models = list((DATA / "encoding/models").rglob("*.safetensors"))
    check(f"encoding safetensors: {len(enc_models)} (expect 16)", len(enc_models) == 16)
    wm_models = list((DATA / "watermark_encoding/models").rglob("*.safetensors"))
    check(f"watermark_encoding safetensors: {len(wm_models)} (expect 8)", len(wm_models) == 8)


# ── 3. no weights in repo working tree ───────────────────────────────────────
section("3. repo working tree has no large artifacts")
pth_in_repo = [f for f in REPO.rglob("*.pth") if ".git" not in str(f)]
safe_in_repo = [f for f in REPO.rglob("*.safetensors") if ".git" not in str(f)]
check(f"no .pth files in repo ({len(pth_in_repo)} found)", len(pth_in_repo) == 0,
      "\n        ".join(str(f) for f in pth_in_repo))
check(f"no .safetensors files in repo ({len(safe_in_repo)} found)", len(safe_in_repo) == 0,
      "\n        ".join(str(f) for f in safe_in_repo))


# ── 4. YAML configs load and expand paths ────────────────────────────────────
section("4. decoder YAML configs")
try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print(f"  {SKIP}  (pyyaml not installed)")

if HAS_YAML and p is not None:
    configs_dir = REPO / "decoding/configs"
    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        with open(cfg_path) as f:
            raw = f.read()
        expanded = raw.replace("${PROJECT_DATA_ROOT}", str(p.data_root))
        try:
            cfg = yaml.safe_load(expanded)
            ckpt = cfg.get("output", {}).get("checkpoint", "")
            meta = cfg.get("data", {}).get("metadata_path", "")
            check(
                f"{cfg_path.name}: checkpoint path resolves",
                "${" not in ckpt and bool(ckpt),
            )
            check(
                f"{cfg_path.name}: metadata path resolves",
                "${" not in meta and bool(meta),
            )
        except Exception as exc:
            check(f"{cfg_path.name}: parses as valid YAML", False, str(exc))


# ── 5. metadata.json structure ───────────────────────────────────────────────
section("5. metadata.json structure")
if p is not None and p.metadata.exists():
    try:
        with open(p.metadata) as f:
            meta = json.load(f)
        check("metadata.json is valid JSON", True)
        check(f"metadata has entries (got {len(meta)})", isinstance(meta, list) and len(meta) > 0)
        if meta:
            first = meta[0]
            check("entries have 'file' key", "file" in first)
            check("entries have 'bits' key (list of 8)", "bits" in first and len(first["bits"]) == 8)
            check("entries have 'id_int' key", "id_int" in first)
    except Exception as exc:
        check("metadata.json loads", False, str(exc))
else:
    print(f"  {SKIP}  (metadata not found)")


# ── 6. dataloader instantiates against real data ─────────────────────────────
section("6. WatermarkDataset instantiation (no GPU)")
if p is not None and p.metadata.exists() and p.images_dir.exists():
    try:
        from decoding.data.dataset import WatermarkDataset
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        ds = WatermarkDataset(
            metadata_path=str(p.metadata),
            image_dir=str(p.images_dir),
            transform=tf,
        )
        check(f"dataset length (got {len(ds)})", len(ds) > 0)
        sample = ds[0]
        check("sample has 'image' tensor", "image" in sample and hasattr(sample["image"], "shape"))
        check("sample has 'bits' tensor of len 8", "bits" in sample and len(sample["bits"]) == 8)
        check("image shape is (3, 64, 64)", tuple(sample["image"].shape) == (3, 64, 64))
    except Exception as exc:
        check("WatermarkDataset instantiates", False, str(exc))
else:
    print(f"  {SKIP}  (metadata or images dir not found)")


# ── 7. checkpoint files are valid PyTorch zip archives ───────────────────────
section("7. checkpoint archive integrity (header only, no full load)")
if p is not None:
    checkpoints_to_probe = [
        p.decoder_checkpoints / "dual_branch_r50.pth",
        p.decoder_checkpoints / "baseline_resnet50.pth",
        p.model_bundle("dual_branch_r50") / "dual_branch_r50.pth",
    ]
    for ckpt in checkpoints_to_probe:
        if not ckpt.exists():
            print(f"  {SKIP}  {ckpt.name} (not found)")
            continue
        try:
            with zipfile.ZipFile(ckpt) as zf:
                names = zf.namelist()
            check(f"{ckpt.parent.name}/{ckpt.name} is valid zip archive ({len(names)} entries)", True)
        except Exception as exc:
            check(f"{ckpt.name} is valid zip archive", False, str(exc))
else:
    print(f"  {SKIP}  (Paths failed)")


# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
if failures:
    print(f"  {FAIL}  {len(failures)} check(s) failed:")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    print(f"  {PASS}  All checks passed.")
print(f"{'═'*60}\n")
