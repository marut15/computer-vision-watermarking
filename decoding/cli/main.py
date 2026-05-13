"""Top-level argparse dispatcher for the decoder CLI."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .commands import clean as cmd_clean
from .commands import decode as cmd_decode
from .commands import export_summaries as cmd_export
from .commands import s3_plan as cmd_s3
from .commands import test_cmd as cmd_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m decoding.cli",
        description="Decoder CLI: test, decode, export-summaries, s3-plan, clean.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_test = sub.add_parser("test", help="evaluate decoder model(s) on a split")
    cmd_test.add_arguments(p_test)
    p_test.set_defaults(func=cmd_test.run)

    p_decode = sub.add_parser("decode", help="decode watermarked images")
    cmd_decode.add_arguments(p_decode)
    p_decode.set_defaults(func=cmd_decode.run)

    p_export = sub.add_parser("export-summaries",
                              help="bundle existing summary artifacts")
    cmd_export.add_arguments(p_export)
    p_export.set_defaults(func=cmd_export.run)

    p_s3 = sub.add_parser("s3-plan", help="emit proposed S3 layout (JSON + MD)")
    cmd_s3.add_arguments(p_s3)
    p_s3.set_defaults(func=cmd_s3.run)

    p_clean = sub.add_parser("clean", help="remove obviously stale local files")
    cmd_clean.add_arguments(p_clean)
    p_clean.set_defaults(func=cmd_clean.run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
