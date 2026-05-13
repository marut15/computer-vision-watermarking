"""Device selection — kept torch-free at import time."""
from __future__ import annotations


def pick_device(name: str = "auto") -> str:
    """Resolve 'auto|cuda|mps|cpu' to a concrete device string.

    torch is imported lazily so the CLI's help/plumbing code stays usable
    in environments that don't have torch installed.
    """
    name = (name or "auto").lower()
    if name in ("cuda", "mps", "cpu"):
        return name
    if name != "auto":
        raise ValueError(f"unknown device {name!r}; expected auto|cuda|mps|cpu")
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
