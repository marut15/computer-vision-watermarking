"""`decoding.cli export-summaries` — bundle existing summaries into one folder."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from .. import DECODING_ROOT
from ..summary_export import collect_summaries


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results-root", default=str(DECODING_ROOT / "results"),
                   help="Decoder results directory.")
    p.add_argument("--scripts-dir", default=str(DECODING_ROOT / "scripts"),
                   help="Where to look for delta_stats.json and similar.")
    p.add_argument("--output", default=None,
                   help="Destination directory. Defaults to "
                        "decoding/exports/<timestamp>/.")
    p.add_argument("--extra", action="append", default=None, metavar="PATH",
                   help="Additional file(s) to bundle; repeat as needed.")
    p.add_argument("--zip", action="store_true",
                   help="Also write <output>.zip alongside the folder.")


def run(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        output = str(DECODING_ROOT / "exports" / ts)
    extras: List[Path] = [Path(p) for p in (args.extra or [])]

    res = collect_summaries(
        results_root=Path(args.results_root),
        output_dir=Path(output),
        scripts_dir=Path(args.scripts_dir) if args.scripts_dir else None,
        extra_paths=extras,
        make_zip=args.zip,
    )
    print(f"output:    {res.output_dir}")
    print(f"collected: {len(res.collected)} files")
    print(f"missing:   {len(res.missing)} files")
    for m in res.missing:
        print(f"  missing: {m}")
    if res.zip_path:
        print(f"zip:       {res.zip_path}")
    return 0
