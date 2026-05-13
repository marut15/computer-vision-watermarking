"""`decoding.cli clean` — safe local cleanup. Dry-run by default."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Set

from .. import DECODING_ROOT, REPO_ROOT


# Directories that are always safe to wipe — never contain source or results.
SAFE_DIR_NAMES: Set[str] = {"__pycache__", ".ipynb_checkpoints"}

# Files that are always safe to delete — caches and OS noise.
SAFE_FILE_NAMES: Set[str] = {".DS_Store", "Thumbs.db"}
SAFE_FILE_SUFFIXES: Set[str] = {".pyc", ".pyo"}

# Files we will only delete with explicit flags.
WEIGHT_SUFFIXES: Set[str] = {".pth", ".pt", ".safetensors"}
FIGURE_SUFFIXES: Set[str] = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

# Never enter (or delete inside) these directories — committed results live here.
PROTECTED_DIRS: Set[str] = {".git", "decoding/configs", "decoding/data"}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=str(REPO_ROOT),
                   help="Repository root to scan (default: repo root).")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Show what would be deleted, but do not delete. Default.")
    p.add_argument("--yes", action="store_true",
                   help="Actually delete. Required to overwrite --dry-run.")
    p.add_argument("--delete-weights", action="store_true",
                   help="Also delete .pth/.pt/.safetensors files. Off by default.")
    p.add_argument("--delete-figures", action="store_true",
                   help="Also delete generated figures. Off by default.")


def _is_protected(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True  # outside the scan root is always protected
    rel_str = str(rel).replace("\\", "/")
    for protected in PROTECTED_DIRS:
        if rel_str == protected or rel_str.startswith(protected + "/"):
            return True
    return False


def find_targets(
    root: Path,
    delete_weights: bool,
    delete_figures: bool,
) -> List[Path]:
    targets: List[Path] = []
    root = root.resolve()
    for path in root.rglob("*"):
        # Skip protected zones entirely.
        if _is_protected(path, root):
            continue
        name = path.name
        if path.is_dir():
            if name in SAFE_DIR_NAMES:
                targets.append(path)
        elif path.is_file():
            if name in SAFE_FILE_NAMES or path.suffix in SAFE_FILE_SUFFIXES:
                targets.append(path)
                continue
            if delete_weights and path.suffix in WEIGHT_SUFFIXES:
                targets.append(path)
                continue
            if delete_figures and path.suffix in FIGURE_SUFFIXES:
                # Only delete figures under decoding/figures/ or results/figures/
                rel = str(path.relative_to(root)).replace("\\", "/")
                if "decoding/figures/" in rel or "decoding/results/figures/" in rel:
                    targets.append(path)
    # Deduplicate (a __pycache__ dir may also contain matching .pyc files).
    seen: Set[Path] = set()
    unique: List[Path] = []
    for t in sorted(targets, key=lambda p: (-len(str(p)), str(p))):
        # Skip files that live under a directory we already plan to delete.
        if any(parent in seen for parent in t.parents):
            continue
        seen.add(t)
        unique.append(t)
    unique.sort()
    return unique


def _delete(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: --root does not exist: {root}")
        return 2
    targets = find_targets(root, args.delete_weights, args.delete_figures)
    if not targets:
        print("nothing to clean.")
        return 0

    actually_delete = args.yes  # --yes overrides default --dry-run
    print(f"=== clean ({'DELETE' if actually_delete else 'dry-run'}) ===")
    print(f"  root: {root}")
    print(f"  flags: weights={args.delete_weights} figures={args.delete_figures}")
    print(f"  {len(targets)} candidate(s):")
    for t in targets:
        kind = "dir " if t.is_dir() else "file"
        print(f"   {kind}  {t.relative_to(root)}")
    if not actually_delete:
        print("\nDry-run: pass --yes to actually delete.")
        return 0

    n_deleted = 0
    for t in targets:
        try:
            _delete(t)
            n_deleted += 1
        except OSError as e:
            print(f"  failed: {t}: {e}")
    print(f"\ndeleted {n_deleted}/{len(targets)} entries.")
    return 0
