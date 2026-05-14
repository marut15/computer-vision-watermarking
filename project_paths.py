"""Canonical path resolver for the computer-vision-watermarking project.

Data lives outside the repo under a configurable root. Set PROJECT_DATA_ROOT
to override; defaults to ../data/computer-vision-watermarking relative to the
repo root so the standard layout /workspace/repo + /workspace/data just works.

Usage:
    from project_paths import Paths
    p = Paths()
    dataset = WatermarkDataset(p.metadata, p.images_dir)
    ckpt = p.checkpoint("dual_branch_r50")

Or with a custom data root:
    p = Paths(data_root="/mnt/storage/cvm-data")
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOT = _REPO_ROOT.parents[1] / "data" / "computer-vision-watermarking"


class Paths:
    def __init__(self, data_root: str | Path | None = None):
        env = os.environ.get("PROJECT_DATA_ROOT")
        if data_root is not None:
            self.data_root = Path(data_root).resolve()
        elif env:
            self.data_root = Path(env).resolve()
        else:
            self.data_root = _DEFAULT_DATA_ROOT.resolve()

        self.repo_root = _REPO_ROOT

    # ── decoder ──────────────────────────────────────────────────────────────
    @property
    def decoder_checkpoints(self) -> Path:
        return self.data_root / "decoding" / "checkpoints"

    @property
    def model_bundles(self) -> Path:
        return self.data_root / "decoding" / "model_bundles"

    def checkpoint(self, name: str) -> Path:
        """Return path to a named checkpoint, e.g. checkpoint('dual_branch_r50')."""
        return self.decoder_checkpoints / f"{name}.pth"

    def model_bundle(self, name: str) -> Path:
        return self.model_bundles / name

    # ── watermark data ───────────────────────────────────────────────────────
    @property
    def watermark_data(self) -> Path:
        return self.data_root / "watermark_encoding" / "data"

    @property
    def images_dir(self) -> Path:
        return self.watermark_data / "images"

    @property
    def baseline_dir(self) -> Path:
        return self.watermark_data / "baseline"

    @property
    def metadata(self) -> Path:
        return self.watermark_data / "metadata.json"

    # ── encoder models ───────────────────────────────────────────────────────
    @property
    def encoder_models(self) -> Path:
        return self.data_root / "encoding" / "models"

    @property
    def watermark_encoder_models(self) -> Path:
        return self.data_root / "watermark_encoding" / "models"

    # ── repo-side (small, tracked) ───────────────────────────────────────────
    @property
    def splits(self) -> Path:
        return self.repo_root / "decoding" / "data" / "splits.json"

    @property
    def decoding_configs(self) -> Path:
        return self.repo_root / "decoding" / "configs"

    def __repr__(self) -> str:
        return f"Paths(data_root={self.data_root})"
