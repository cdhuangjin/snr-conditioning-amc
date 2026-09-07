from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


sha256_file = file_sha256


def deterministic_json_hash(value: Any) -> str:
    """Hash JSON/config data using a canonical key order and compact encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_fingerprint() -> dict[str, object]:
    """Capture lightweight runtime and importable scientific-package versions."""
    packages: dict[str, str] = {}
    candidates = {
        "numpy": "numpy",
        "scipy": "scipy",
        "torch": "torch",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
    }
    for module_name, distribution_name in candidates.items():
        if importlib.util.find_spec(module_name) is None:
            continue
        try:
            packages[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
