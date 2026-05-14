"""Checkpoint discovery + loading.

Two layouts are supported simultaneously:

1. `--weights-root /workspace/new_models` (the S3 staging layout)
   <root>/<arch>/<arch>.pth          (e.g. new_models/dual_branch_r50/dual_branch_r50.pth)

2. `--weights-root decoding/checkpoints` (the legacy repo layout)
   <root>/<arch>.pth                  (e.g. decoding/checkpoints/baseline_resnet50.pth)
   <root>/<legacy_filename>.pth       (matches ModelSpec.legacy_checkpoint)

Per-model `--checkpoint <arch>=<path>` overrides win over both.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .registry import ModelSpec


@dataclass(frozen=True)
class ResolvedCheckpoint:
    spec: ModelSpec
    path: Optional[Path]    # None if no candidate exists locally
    candidates_tried: Tuple[Path, ...]


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Path]:
    """Parse --checkpoint NAME=PATH pairs."""
    out: Dict[str, Path] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(
                f"--checkpoint expects NAME=PATH, got {raw!r}"
            )
        name, _, path = raw.partition("=")
        out[name.strip().lower()] = Path(path).expanduser()
    return out


def candidate_paths(spec: ModelSpec, weights_root: Path) -> List[Path]:
    """Generate the search order for a model's checkpoint under weights_root."""
    weights_root = Path(weights_root).expanduser()
    candidates: List[Path] = []
    # S3 staging layout: <root>/<arch>/<arch>.pth
    candidates.append(weights_root / spec.name / spec.checkpoint_filename)
    # Flat layout: <root>/<arch>.pth
    candidates.append(weights_root / spec.checkpoint_filename)
    # Legacy filename in same root (only if it differs)
    if spec.legacy_checkpoint and spec.legacy_checkpoint != spec.checkpoint_filename:
        candidates.append(weights_root / spec.legacy_checkpoint)
    # Some bundles drop the file directly named `model.pth`
    candidates.append(weights_root / spec.name / "model.pth")
    return candidates


def resolve_checkpoint(
    spec: ModelSpec,
    weights_root: Path,
    overrides: Optional[Dict[str, Path]] = None,
) -> ResolvedCheckpoint:
    overrides = overrides or {}
    if spec.name in overrides:
        chosen = overrides[spec.name].expanduser()
        return ResolvedCheckpoint(spec=spec, path=chosen, candidates_tried=(chosen,))
    cands = candidate_paths(spec, weights_root)
    for c in cands:
        if c.is_file():
            return ResolvedCheckpoint(spec=spec, path=c, candidates_tried=tuple(cands))
    return ResolvedCheckpoint(spec=spec, path=None, candidates_tried=tuple(cands))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
