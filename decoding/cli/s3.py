"""Optional S3 download helpers.

boto3 is imported lazily — the rest of the CLI must keep working without
AWS credentials installed. Only the `decode --s3-uri ... --download-missing`
path touches the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Return (bucket, key) for an s3://bucket/key/path uri."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri!r}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"missing bucket in {uri!r}")
    return bucket, key


def join_s3(uri: str, *parts: str) -> str:
    base = uri.rstrip("/")
    extras = "/".join(p.strip("/") for p in parts if p)
    return f"{base}/{extras}" if extras else base


def download_s3_object(uri: str, dest: Path) -> Path:
    """Download a single S3 object to `dest`. Lazy-imports boto3."""
    import boto3  # type: ignore[import-not-found]
    bucket, key = parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    client.download_file(bucket, key, str(dest))
    return dest
