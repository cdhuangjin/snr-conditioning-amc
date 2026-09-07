from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from secrets import token_hex
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from v2.progress import update_phase_progress
from v2.manifest import _cross_process_lock


DEFAULT_MAX_HASH_BYTES = 128 * 1024 * 1024
SCANNER_VERSION = "phase0-audit/3"
HISTORICAL_RESULT_VALIDATION_VERSION = "historical-result/v1"
RECOGNIZED_RESULT_METRICS = frozenset(
    {
        "acc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "loss",
        "macro_f1",
        "macro_f1_score",
        "overall",
        "overall_accuracy",
    }
)
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
SCAN_DIRECTORIES = (
    "checkpoint",
    "training",
    "experiments",
    "data",
    "downloads",
    "models",
    "data_loader",
    "configs",
    "config",
    "splits",
)
SCRIPT_SUFFIXES = {".py", ".ps1", ".sh", ".bat", ".cmd"}
ROOT_SCRIPT_PREFIXES = (
    "_exp",
    "experiment",
    "run",
    "plot",
    "train",
    "eval",
    "summarize",
    "bench",
    "generate",
    "print_paper",
)
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".eps", ".tif", ".tiff"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".pkl", ".h5", ".hdf5"}
RESULT_SUFFIXES = {".json", ".npz", ".npy", ".csv", ".tsv"}
LOG_SUFFIXES = {".log", ".out", ".err"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".ini", ".cfg"}


class AuditSecurityError(RuntimeError):
    """Raised when containment or reparse-point safety cannot be guaranteed."""


@dataclass
class ScanResult:
    assets: list[dict[str, object]]
    issues: list[dict[str, str]]

    @property
    def complete(self) -> bool:
        return not self.issues


class _NamedBinaryHandle:
    def __init__(self, handle: BinaryIO, path: Path):
        self._handle = handle
        self.name = str(path)

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def fileno(self) -> int:
        return self._handle.fileno()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_NamedBinaryHandle":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _fdopen_owned_descriptor(
    descriptor: int,
    path: Path,
    *,
    fdopen: Callable[..., BinaryIO] = os.fdopen,
    close_descriptor: Callable[[int], object] = os.close,
) -> _NamedBinaryHandle:
    """Convert an owned descriptor to a file object without leaking it on failure."""
    try:
        handle = fdopen(descriptor, "rb", closefd=True)
    except BaseException as primary:
        try:
            close_descriptor(descriptor)
        except BaseException as close_error:
            primary.add_note(
                f"descriptor cleanup also failed: {close_error.__class__.__name__}: {close_error}"
            )
        raise
    return _NamedBinaryHandle(handle, path)


def _sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _normalized(path: str | os.PathLike[str]) -> str:
    normalized = str(path).replace("\\", "/")
    return normalized[2:] if normalized.startswith("./") else normalized


def classify_path(path: str | os.PathLike[str]) -> str:
    """Classify a repository-relative asset path without reading its content."""
    normalized = _normalized(path)
    lowered = normalized.casefold()
    parts = PurePosixPath(lowered).parts
    suffix = PurePosixPath(lowered).suffix
    name = PurePosixPath(lowered).name

    if "splits" in parts or (
        suffix in {".json", ".npz", ".npy"}
        and (name.startswith("split_") or name.endswith("_split.json") or name.endswith("_split.npz"))
    ):
        return "split_file"
    if parts and (parts[0] in {"data", "downloads"} or lowered.startswith("../triple/data/")):
        return "dataset"
    if suffix in FIGURE_SUFFIXES:
        return "figure"
    if (
        "data_loader" in parts
        or "preprocessing" in parts
        or "preprocess" in name
        or "data_loader" in name
    ) and suffix in SCRIPT_SUFFIXES:
        return "preprocessing_pipeline"
    if suffix in CONFIG_SUFFIXES or (parts and parts[0] in {"config", "configs"}):
        return "config"
    if suffix in LOG_SUFFIXES or any(part in {"log", "logs"} for part in parts):
        return "log"
    if suffix in CHECKPOINT_SUFFIXES and (
        parts and parts[0] == "checkpoint" or any(part in {"model", "models"} for part in parts)
    ):
        return "checkpoint"
    if suffix in SCRIPT_SUFFIXES and (
        "experiments" in parts
        or "training" in parts
        or name.startswith(ROOT_SCRIPT_PREFIXES)
    ):
        return "experiment_script"
    if suffix in RESULT_SUFFIXES and (
        "experiments" in parts or (len(parts) >= 2 and parts[:2] == ("triple", "results"))
        or normalized.startswith("../triple/results/")
    ):
        return "historical_result"
    return "unknown"


def _is_reparse_stat(file_stat: os.stat_result) -> bool:
    attributes = int(getattr(file_stat, "st_file_attributes", 0))
    return stat.S_ISLNK(file_stat.st_mode) or bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def _lexists(path: Path) -> bool:
    try:
        _lstat(path)
    except FileNotFoundError:
        return False
    return True


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _security_error(path: Path, message: str) -> AuditSecurityError:
    return AuditSecurityError(f"{path}: reparse/containment safety failure: {message}")


def _checked_directory(path: Path, allowed_parent: Path) -> Path | None:
    try:
        file_stat = _lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _security_error(path, str(exc)) from exc
    if _is_reparse_stat(file_stat):
        raise _security_error(path, "directory is a symlink, junction, or reparse point")
    if not stat.S_ISDIR(file_stat.st_mode):
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _security_error(path, f"cannot resolve directory: {exc}") from exc
    if not _is_within(resolved, allowed_parent):
        raise _security_error(path, f"resolved outside {allowed_parent}")
    return path


def _scan_roots(root: Path) -> list[Path]:
    roots = []
    for name in SCAN_DIRECTORIES:
        checked = _checked_directory(root / name, root)
        if checked is not None:
            roots.append(checked)
    repository_parent = root.parent.resolve(strict=True)
    triple = root.parent / "triple"
    triple_results = triple / "results"
    checked_triple = _checked_directory(triple, repository_parent)
    if checked_triple is not None:
        checked_results = _checked_directory(triple_results, repository_parent)
        if checked_results is not None:
            roots.append(checked_results)
    return roots


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def _issue(path: str, error_kind: str, message: object) -> dict[str, str]:
    return {"path": path, "error_kind": error_kind, "message": str(message)}


def _open_binary_no_follow(path: Path) -> _NamedBinaryHandle:
    if os.name == "nt":
        import ctypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x80000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x00200000 | 0x08000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            error = ctypes.get_last_error()
            if error in {2, 3}:
                raise FileNotFoundError(error, os.strerror(error), str(path))
            if error == 5:
                raise PermissionError(error, os.strerror(error), str(path))
            raise OSError(error, os.strerror(error), str(path))
        try:
            descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except BaseException:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            raise
        return _fdopen_owned_descriptor(descriptor, path)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    return _fdopen_owned_descriptor(descriptor, path)


def _stat_signature(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
    )


def _normalized_protocol_hash(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return value.casefold()


def _is_finite_numeric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _evidence_state(
    relative: str,
    asset_type: str,
    content: bytes | None,
    expected_protocol_hash: str | None = None,
) -> tuple[str, bool, str]:
    if asset_type != "historical_result":
        return "candidate artifact", False, "Candidate artifact; semantics require phase-specific review."
    if PurePosixPath(relative.casefold()).suffix != ".json" or content is None:
        return (
            "unverified historical result",
            False,
            "Unverified historical result candidate; filename and location do not validate its contents.",
        )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            "unverified historical result",
            False,
            "Unverified historical result candidate; JSON is malformed or not UTF-8.",
        )
    if not isinstance(payload, dict):
        return (
            "unverified historical result",
            False,
            "Unverified historical result candidate; JSON root is not an object.",
        )
    status_ok = payload.get("status") == "completed"
    metric_values: list[object] = [
        value for key, value in payload.items() if key in RECOGNIZED_RESULT_METRICS
    ]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metric_values.extend(
            value for key, value in metrics.items() if key in RECOGNIZED_RESULT_METRICS
        )
    metrics_ok = any(_is_finite_numeric(value) for value in metric_values)
    if not status_ok or not metrics_ok:
        return (
            "unverified historical result",
            False,
            f"Unverified historical result candidate under {HISTORICAL_RESULT_VALIDATION_VERSION}; "
            "status must be exactly completed and a recognized metric must be a finite real number.",
        )
    protocol_hash = payload.get("protocol_hash")
    actual_hash = None
    if isinstance(protocol_hash, str):
        try:
            actual_hash = _normalized_protocol_hash(protocol_hash, field_name="protocol_hash")
        except ValueError:
            pass
    if actual_hash is None:
        return (
            "unverified historical result",
            False,
            f"Unverified historical result candidate under {HISTORICAL_RESULT_VALIDATION_VERSION}; "
            "protocol_hash must be a full 64-character hexadecimal SHA-256.",
        )
    if expected_protocol_hash is not None and actual_hash == expected_protocol_hash:
        return (
            "validated machine-readable result",
            True,
            f"Validated by {HISTORICAL_RESULT_VALIDATION_VERSION}: completed status, finite recognized "
            "metric, and trusted matching protocol SHA-256.",
        )
    if expected_protocol_hash is None:
        protocol_note = "no trusted expected protocol SHA-256 was supplied"
    else:
        protocol_note = "protocol_hash does not match the trusted expected SHA-256"
    return (
        "validated historical result candidate",
        False,
        f"Validated structural candidate under {HISTORICAL_RESULT_VALIDATION_VERSION}, but {protocol_note}; "
        "scientific reuse still requires protocol-hash review.",
    )


def _reuse_for(
    path: str, asset_type: str, evidence_state: str, protocol_verified: bool
) -> tuple[str, str]:
    lowered = path.casefold()
    if lowered.startswith("experiments/v2/"):
        return (
            "do_not_reuse",
            "Legacy experiments/v2 asset; keep read-only and do not mix it with the new results/v2 namespace.",
        )
    if asset_type == "historical_result" and protocol_verified:
        return (
            "reuse_as_is",
            "Validated machine-readable result with an explicit protocol hash; preserve read-only and never present it as a V2 run.",
        )
    if asset_type == "historical_result":
        return (
            "reuse_after_protocol_hash_check",
            f"{evidence_state}; preserve read-only but require protocol and content-hash review before reuse.",
        )
    if asset_type in {"checkpoint", "dataset", "split_file"}:
        return (
            "reuse_after_protocol_hash_check",
            "Reuse only after content hash, split, preprocessing, and protocol compatibility are verified.",
        )
    if asset_type in {"figure", "experiment_script", "preprocessing_pipeline", "config", "log"}:
        return (
            "reference_only",
            "Use for provenance or protocol reconstruction; regenerate V2 evidence from persisted V2 outputs.",
        )
    return (
        "do_not_reuse",
        "Unclassified asset; retain in place but exclude until its provenance and semantics are established.",
    )


def _placeholder_record(relative: str, marker: str, message: str) -> dict[str, object]:
    asset_type = classify_path(relative)
    evidence_state, protocol_verified, evidence_note = _evidence_state(relative, asset_type, None)
    reuse, reuse_note = _reuse_for(relative, asset_type, evidence_state, protocol_verified)
    return {
        "path": relative,
        "asset_type": asset_type,
        "size_bytes": "unknown",
        "mtime": "unknown",
        "sha256": f"not_computed:{marker}",
        "evidence_state": evidence_state,
        "reuse": reuse,
        "notes": f"{evidence_note} {reuse_note} Scan issue: {message}",
    }


def _scan_file(
    path: Path,
    relative: str,
    max_hash_bytes: int,
    chunk_size: int,
    open_file: Callable[[Path], BinaryIO],
    read_chunk: Callable[[BinaryIO, int], bytes],
    event_hook: Callable[[str, Path, int], None] | None,
    expected_protocol_hash: str | None,
) -> tuple[dict[str, object], dict[str, str] | None]:
    try:
        before = _lstat(path)
    except FileNotFoundError as exc:
        return _placeholder_record(relative, "vanished", str(exc)), _issue(relative, "vanished", exc)
    except OSError as exc:
        return _placeholder_record(relative, "unreadable", str(exc)), _issue(relative, "unreadable", exc)
    if _is_reparse_stat(before):
        raise _security_error(path, "asset is a symlink or reparse point")
    if not stat.S_ISREG(before.st_mode):
        message = "candidate is not a regular file"
        return _placeholder_record(relative, "unreadable", message), _issue(relative, "unreadable", message)

    marker: str | None = None
    issue: dict[str, str] | None = None
    digest = hashlib.sha256()
    content = bytearray() if PurePosixPath(relative.casefold()).suffix == ".json" else None
    total = 0
    after_handle = before
    try:
        with open_file(path) as handle:
            opened = os.fstat(handle.fileno())
            if _stat_signature(opened) != _stat_signature(before):
                marker = "unstable_during_scan"
                issue = _issue(relative, marker, "metadata changed between lstat and open")
            elif opened.st_size > max_hash_bytes:
                marker = "size_limit"
            else:
                while marker is None:
                    requested = min(chunk_size, max_hash_bytes - total + 1)
                    try:
                        chunk = read_chunk(handle, requested)
                    except OSError as exc:
                        marker = "hash_error"
                        issue = _issue(relative, marker, exc)
                        break
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_hash_bytes:
                        marker = "size_limit"
                        break
                    digest.update(chunk)
                    if content is not None:
                        content.extend(chunk)
                    if event_hook is not None:
                        event_hook("after_chunk", path, total)
            after_handle = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        marker = "vanished"
        issue = _issue(relative, marker, exc)
    except PermissionError as exc:
        marker = "unreadable"
        issue = _issue(relative, marker, exc)
    except OSError as exc:
        marker = "unreadable"
        issue = _issue(relative, marker, exc)

    if event_hook is not None:
        event_hook("before_post_lstat", path, total)
    try:
        after_path = _lstat(path)
    except FileNotFoundError as exc:
        marker = "vanished"
        issue = _issue(relative, marker, exc)
        after_path = after_handle
    except OSError as exc:
        marker = "unreadable"
        issue = _issue(relative, marker, exc)
        after_path = after_handle
    else:
        if _is_reparse_stat(after_path):
            raise _security_error(path, "asset became a symlink or reparse point during scan")
        if (
            _stat_signature(after_handle) != _stat_signature(before)
            or _stat_signature(after_path) != _stat_signature(after_handle)
        ):
            if marker != "vanished":
                issue = _issue(relative, "unstable_during_scan", "identity or metadata changed during scan")
                if marker != "size_limit":
                    marker = "unstable_during_scan"

    asset_type = classify_path(relative)
    stable_content = bytes(content) if content is not None and marker is None else None
    evidence_state, protocol_verified, evidence_note = _evidence_state(
        relative, asset_type, stable_content, expected_protocol_hash
    )
    reuse, reuse_note = _reuse_for(relative, asset_type, evidence_state, protocol_verified)
    marker_value = f"not_computed:{marker}" if marker else digest.hexdigest()
    notes = f"{evidence_note} {reuse_note}"
    if issue is not None:
        notes += f" Scan issue [{issue['error_kind']}]: {issue['message']}"
    return (
        {
            "path": relative,
            "asset_type": asset_type,
            "size_bytes": int(after_handle.st_size),
            "mtime": datetime.fromtimestamp(after_handle.st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "sha256": marker_value,
            "evidence_state": evidence_state,
            "reuse": reuse,
            "notes": notes,
        },
        issue,
    )


def _discover_candidates(repository: Path) -> tuple[dict[str, Path], list[dict[str, str]]]:
    candidates: dict[str, Path] = {}
    issues: list[dict[str, str]] = []

    def walk_error(exc: OSError) -> None:
        path = _normalized(getattr(exc, "filename", "unknown"))
        issues.append(_issue(path, "walk_error", exc))

    for scan_root in _scan_roots(repository):
        for directory, directory_names, file_names in os.walk(
            scan_root, topdown=True, followlinks=False, onerror=walk_error
        ):
            current = Path(directory)
            safe_directories = []
            for name in sorted(directory_names, key=_sort_key):
                candidate = current / name
                try:
                    child_stat = _lstat(candidate)
                except FileNotFoundError as exc:
                    relative = _relative_path(candidate, repository).rstrip("/") + "/"
                    issues.append(_issue(relative, "walk_vanished", exc))
                    continue
                except OSError as exc:
                    issues.append(_issue(_relative_path(candidate, repository), "walk_error", exc))
                    continue
                if _is_reparse_stat(child_stat):
                    raise _security_error(candidate, "scan directory is a symlink or reparse point")
                if stat.S_ISDIR(child_stat.st_mode):
                    safe_directories.append(name)
            directory_names[:] = safe_directories
            for name in sorted(file_names, key=_sort_key):
                candidate = current / name
                try:
                    child_stat = _lstat(candidate)
                except FileNotFoundError:
                    candidates[_relative_path(candidate, repository)] = candidate
                    continue
                except OSError as exc:
                    relative = _relative_path(candidate, repository)
                    candidates[relative] = candidate
                    issues.append(_issue(relative, "walk_error", exc))
                    continue
                if _is_reparse_stat(child_stat):
                    raise _security_error(candidate, "scan asset is a symlink or reparse point")
                if stat.S_ISREG(child_stat.st_mode):
                    candidates[_relative_path(candidate, repository)] = candidate

    try:
        root_entries = list(os.scandir(repository))
    except OSError as exc:
        issues.append(_issue(".", "walk_error", exc))
        root_entries = []
    for entry in sorted(root_entries, key=lambda value: _sort_key(value.name)):
        path = Path(entry.path)
        try:
            entry_stat = _lstat(path)
        except OSError as exc:
            issues.append(_issue(entry.name, "walk_error", exc))
            continue
        if _is_reparse_stat(entry_stat):
            if entry.name.casefold().startswith(ROOT_SCRIPT_PREFIXES):
                raise _security_error(path, "root script is a symlink or reparse point")
            continue
        if (
            stat.S_ISREG(entry_stat.st_mode)
            and path.suffix.casefold() in SCRIPT_SUFFIXES
            and path.name.casefold().startswith(ROOT_SCRIPT_PREFIXES)
        ):
            candidates[_relative_path(path, repository)] = path
    return candidates, issues


def scan_repository(
    root: str | os.PathLike[str],
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
    *,
    chunk_size: int = 1024 * 1024,
    open_file: Callable[[Path], BinaryIO] | None = None,
    read_chunk: Callable[[BinaryIO, int], bytes] | None = None,
    event_hook: Callable[[str, Path, int], None] | None = None,
    expected_protocol_hash: str | None = None,
) -> ScanResult:
    if isinstance(max_hash_bytes, bool) or not isinstance(max_hash_bytes, int) or max_hash_bytes < 0:
        raise ValueError("max_hash_bytes must be a non-negative integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    normalized_expected_hash = _normalized_protocol_hash(
        expected_protocol_hash, field_name="expected_protocol_hash"
    )
    repository = Path(root).resolve(strict=True)
    if not repository.is_dir():
        raise ValueError(f"repository root is not a directory: {repository}")
    candidates, issues = _discover_candidates(repository)
    open_function = open_file or _open_binary_no_follow
    read_function = read_chunk or (lambda handle, size: handle.read(size))
    assets: list[dict[str, object]] = []
    for relative in sorted(candidates, key=_sort_key):
        record, file_issue = _scan_file(
            candidates[relative],
            relative,
            max_hash_bytes,
            chunk_size,
            open_function,
            read_function,
            event_hook,
            normalized_expected_hash,
        )
        assets.append(record)
        if file_issue is not None:
            issues.append(file_issue)
    for discovery_issue in issues:
        if discovery_issue["error_kind"] == "walk_vanished":
            assets.append(
                _placeholder_record(
                    discovery_issue["path"],
                    discovery_issue["error_kind"],
                    discovery_issue["message"],
                )
            )
    assets.sort(key=lambda item: _sort_key(str(item["path"])))
    issues.sort(key=lambda item: (_sort_key(item["path"]), item["error_kind"], item["message"]))
    return ScanResult(assets=assets, issues=issues)


def scan_assets(
    root: str | os.PathLike[str],
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
    *,
    expected_protocol_hash: str | None = None,
) -> list[dict[str, object]]:
    """Compatibility wrapper returning only assets from a complete or incomplete scan."""
    return scan_repository(
        root,
        max_hash_bytes=max_hash_bytes,
        expected_protocol_hash=expected_protocol_hash,
    ).assets


def _markdown(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("`", "&#96;")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def _code_path(value: object) -> str:
    normalized = str(value).replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")
    escaped = html.escape(normalized, quote=False).replace("`", "&#96;")
    escaped = escaped.replace("|", "\\|")
    return f"<code>{escaped}</code>"


def _report_opening(title: str, metadata: str) -> list[str]:
    lines = [f"# {title}", ""]
    if metadata:
        lines.extend([metadata.rstrip(), ""])
    return lines


def _inventory_report(
    assets: list[dict[str, object]],
    max_hash_bytes: int,
    metadata: str = "",
    issues: list[dict[str, str]] | None = None,
) -> str:
    issues = issues or []
    counts = Counter(str(item["asset_type"]) for item in assets)
    skipped = sum(item["sha256"] == "not_computed:size_limit" for item in assets)
    lines = _report_opening("V2 Phase 0 Asset Inventory", metadata) + [
        "## Evidence summary",
        "",
        f"- Audited assets: {len(assets)}.",
        f"- Asset types: {', '.join(f'{name}={counts[name]}' for name in sorted(counts)) or 'none'}.",
        f"- SHA-256 size limit: {max_hash_bytes} bytes; oversized markers: {skipped}.",
        f"- Scan status: {'complete' if not issues else 'incomplete'}; skipped/error count: {len(issues)}.",
        "- Scope: approved AWN asset directories, root experiment/run/plot scripts, and the resolved sibling ../triple/results directory when present.",
        "- Phase 0 performed no training and no inference; it only read metadata/content for inventory hashing and wrote reports.",
    ]
    if issues:
        lines.extend(
            [
                "",
                "## Scan issues",
                "",
                "| path | error kind | message |",
                "|---|---|---|",
            ]
        )
        for item in issues:
            lines.append(
                "| "
                + " | ".join(_markdown(item[key]) for key in ("path", "error_kind", "message"))
                + " |"
            )
    for asset_type in sorted(counts, key=_sort_key):
        lines.extend(
            [
                "",
                f"## {asset_type}",
                "",
                "| path | asset type | evidence state | bytes | mtime | sha256 | reuse decision | risk/notes |",
                "|---|---|---|---:|---|---|---|---|",
            ]
        )
        for item in assets:
            if item["asset_type"] != asset_type:
                continue
            lines.append(
                "| "
                + " | ".join(
                    _markdown(item[key])
                    for key in (
                        "path",
                        "asset_type",
                        "evidence_state",
                        "size_bytes",
                        "mtime",
                        "sha256",
                        "reuse",
                        "notes",
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def _find_evidence(assets: list[dict[str, object]], *tokens: str, limit: int = 4) -> str:
    matches: list[dict[str, object]] = []
    for item in assets:
        lowered = str(item["path"]).casefold()
        if any(token.casefold() in lowered for token in tokens):
            matches.append(item)
    unique: list[dict[str, object]] = []
    seen = set()
    for item in matches:
        path = str(item["path"])
        if path not in seen:
            unique.append(item)
            seen.add(path)
    if not unique:
        return "not found in audited assets"
    rendered = []
    for item in unique[:limit]:
        state = str(item["evidence_state"])
        label = state if state != "candidate artifact" else "candidate/reference"
        rendered.append(f"{label}: {_code_path(item['path'])}")
    return ", ".join(rendered)


def _issue_rows(assets: list[dict[str, object]]) -> list[tuple[str, str, str, str, str, str]]:
    evidence = lambda *tokens: _find_evidence(assets, *tokens)
    universal_note = "Historical assets are scope-limited; no audited result can establish this universal claim."
    return [
        (
            "Forbidden universal claim: SNR-agnostic classifiers inevitably collapse",
            f"{evidence('base_', 'generic/resnet', 'generic/cldnn')}; {universal_note}",
            "Architecture- and dataset-stratified non-collapse/collapse tests.",
            "Add empirical concentration, balanced-accuracy, and geometry metrics; scan manuscript claims.",
            "Phases 2, 4, 5, 12",
            "Retain counterexamples; prohibit universal wording.",
        ),
        (
            "Forbidden universal claim: loss reweighting is always ineffective",
            f"{evidence('focal', 'snrw')}; {universal_note}",
            "Fixed-split five-seed paired reweighting comparison.",
            "Persist protocol-matched reweighting runs and paired statistics.",
            "Phases 1-3",
            "Claim only observed effects under tested protocols.",
        ),
        (
            "Forbidden universal claim: external SNR is universally necessary",
            f"{evidence('snrc', '_snr_', 'snr_results')}; {universal_note}",
            "Plain-versus-conditioned tests with true, estimated, shuffled, and unavailable SNR.",
            "Implement unified conditioners and E1-E4 controls.",
            "Phases 3, 8-9",
            "A non-collapse or competitive plain model refutes necessity.",
        ),
        (
            "Forbidden universal claim: embedding implements arbitrary per-SNR affine boundaries",
            f"{evidence('snrc', 'model_snr', 'estaware')}; {universal_note}",
            "Capacity-matched embedding, FiLM, gating, and per-bin-head comparison.",
            "Expose feature weights and per-bin biases; document the embedding boundary limit.",
            "Phase 3",
            "No arbitrary-boundary claim from concatenation alone.",
        ),
        (
            "Forbidden universal claim: discrete embedding is inherently continuous",
            f"{evidence('snrc', 'estaware')}; {universal_note}",
            "Interpolation and unseen-condition tests between discrete SNR bins.",
            "Persist bin definitions and test discontinuities explicitly.",
            "Phases 3, 9",
            "Continuity requires empirical evidence, not architecture naming.",
        ),
        (
            "Forbidden universal claim: -8 dB is a universal phase transition",
            f"{evidence('phase_transition', 'reliability', '-8db')}; {universal_note}",
            "Dataset/backbone/seed sensitivity of transition-like diagnostics.",
            "Use reliability-floor conditioning terminology and configurable floors.",
            "Phases 4, 10-12",
            "Treat -8 dB as a tested setting, never a universal threshold.",
        ),
        (
            "Fixed persisted splits and five seeds",
            f"{evidence('multiseed', 'seed2022', 'seed2023')}; historical runs exist, but fixed split identity is not established by filenames.",
            "Persist 60/20/20 class-by-SNR splits for seeds 2022-2026 and reproduce two historical baselines.",
            "Add split IDs, hashes, disjointness checks, and reproduction runner.",
            "Phase 1",
            "Stop if historical accuracy differs by more than 0.5 percentage points.",
        ),
        (
            "Empirical feature geometry and diagnostics",
            evidence('spectral', 'entropy', 'precursors'),
            "Five-seed effective rank, top singular share, separability, entropy, and concentration analysis.",
            "Add centralized diagnostic metrics and persist per-SNR values.",
            "Phases 2, 4",
            "Report association only; no automatic causal interpretation.",
        ),
        (
            "Symmetric non-collapse counterexample and other toy cases",
            evidence('A_asymmetry', 'B_gain', 'C_sharpening'),
            "Symmetric no-collapse, asymmetric collapse, gain, and sharpening controls with fixed seeds.",
            "Implement four reproducible toy generators and persist JSON/NPZ outputs.",
            "Phase 5",
            "Any symmetric non-collapse case blocks inevitability claims.",
        ),
        (
            "Preprocessing leakage and SNR inferability",
            evidence('data_loader', 'preprocess', 'config/'),
            "Train-only, per-frame, and legacy preprocessing audit plus SNR probes.",
            "Implement provenance-tracked preprocessing and probe controls.",
            "Phase 6",
            "Stop and restart Phase 1 if leakage invalidates prior results.",
        ),
        (
            "WBFM dominance analysis",
            evidence('WBFM', 'confmat', 'conf_mat'),
            "Class-frequency, error dominance, signal-statistic, and removal/sensitivity analyses.",
            "Add WBFM diagnostics that operate on persisted predictions and raw samples.",
            "Phase 7",
            "Separate dataset composition from model-collapse evidence.",
        ),
        (
            "Estimator side-channel controls E1-E4",
            evidence('estaware', 'pilot', 'D_estaware', 'E_saturation'),
            "E1 true SNR, E2 estimated SNR, E3 shuffled estimates, and E4 estimate-only prediction.",
            "Implement estimator, shuffled-control, and estimate-only probe APIs.",
            "Phase 8",
            "Conditioning evidence must survive side-channel controls.",
        ),
        (
            "Deployment-matched estimator noise",
            evidence('robust', 'pilot', 'estaware'),
            "Five-seed generic-noise versus deployment-matched estimator-noise training.",
            "Persist estimator error distributions and train with matched sampled errors.",
            "Phase 9",
            "Limit conclusions to tested estimator and datasets.",
        ),
        (
            "Reliability-floor conditioning and non-universal -8 dB terminology",
            evidence('reliability', 'underclamp', 'envelope'),
            "No-clamp versus floors -10, -8, and -6 dB on identical checkpoints/bins.",
            "Implement configurable floor transform and rename all outputs consistently.",
            "Phase 10",
            "No universal-transition terminology; retain all floor outcomes.",
        ),
        (
            "RML2016.04C cross-dataset protocol",
            evidence('04c', '2016.04C'),
            "Leakage-free fixed-split five-seed AWN/plain and selected conditioner runs.",
            "Add dataset adapter, protocol hash, and persisted cross-dataset results.",
            "Phase 11",
            "Protocol compatibility must be explicit before reuse.",
        ),
        (
            "RML2018.01a cross-dataset protocol",
            evidence('2018a', '2018.01a'),
            "Leakage-free fixed-split five-seed protocol on RML2018.01a.",
            "Add HDF5 adapter, class mapping, protocol hash, and persisted results.",
            "Phase 11",
            "Do not merge incompatible label/channel protocols.",
        ),
        (
            "Independent synthetic benchmark",
            evidence('synthetic'),
            "Independent channel-controlled AMC generation and five-seed backbone matrix.",
            "Implement generator, manifest, fixed split, and persisted waveforms/results.",
            "Phase 12",
            "Non-collapse is a valid result and constrains generality.",
        ),
        (
            "Persisted JSON/NPZ-only figures and tables",
            evidence('.json', '.npz'),
            "Regenerate every V2 figure/table solely from registered machine-readable outputs.",
            "Add plotting/report builders that reject hand-entered experiment numbers.",
            "Phases 3-12, final",
            "Every displayed number resolves to a persisted artifact field.",
        ),
        (
            "Negative result retention",
            evidence('run_err', '.err', '.log'),
            "Register scientifically valid null/negative outcomes distinctly from failed execution.",
            "Keep completed negative results in manifest and progress reports.",
            "All phases",
            "Never delete or relabel a negative result as a failed run.",
        ),
        (
            "Paired statistics",
            evidence('stats', 'summary'),
            "Five-seed paired differences, 95% t intervals, effect sizes, and secondary p-values.",
            "Add centralized paired-summary implementation and JSON schema.",
            "Phases 2-12",
            "Statistics must use matched seeds and fixed split hashes.",
        ),
        (
            "Manifest, progress, and final deliverables",
            evidence('result.json', 'summary.json'),
            "Artifact-completeness audit tying issue, code, run, figure, table, and manuscript text.",
            "Use manifest/progress APIs and build final reviewer/deliverable maps.",
            "Phase 0 and final",
            "Final gate fails on missing hashes, outputs, negative results, or forbidden claims.",
        ),
    ]


def _reviewer_report(assets: list[dict[str, object]], metadata: str = "") -> str:
    lines = _report_opening("V2 Reviewer Issue Matrix", metadata) + [
        "This is an evidence map, not a claim that historical assets already satisfy a V2 gate.",
        "",
        "| issue | existing evidence | missing experiment | required code change | phase | gate |",
        "|---|---|---|---|---|---|",
    ]
    for row in _issue_rows(assets):
        lines.append("| " + " | ".join(_markdown(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _target_phase(asset_type: str) -> str:
    return {
        "historical_result": "Phase 1 reproduction and later evidence comparison",
        "checkpoint": "Phase 1 reproduction after protocol verification",
        "dataset": "Phases 1, 6, 11, or 12 after hash verification",
        "split_file": "Phase 1 only after split contract verification",
        "figure": "Phase 0 provenance only",
        "experiment_script": "Phase 0 protocol reconstruction only",
        "preprocessing_pipeline": "Phase 6 leakage audit",
        "config": "Phase 0 protocol reconstruction only",
        "log": "Phase 0 provenance and failure reconstruction",
        "unknown": "None until manually classified",
    }[asset_type]


def _reuse_report(assets: list[dict[str, object]], metadata: str = "") -> str:
    lines = _report_opening("V2 Historical Asset Reuse Map", metadata) + [
        "Preserve all historical result candidates in place as read-only artifacts; validation state controls reuse and does not certify scientific conclusions.",
        "The directory experiments/v2 is legacy and is distinct from the new results/v2 namespace; legacy V2-named assets must not be treated as outputs of this rebuild.",
        "",
        "Allowed decisions: `reuse_as_is`, `reuse_after_protocol_hash_check`, `reference_only`, `do_not_reuse`.",
        "",
        "| path | asset type | evidence state | decision | reason | target V2 phase |",
        "|---|---|---|---|---|---|",
    ]
    for item in assets:
        lines.append(
            "| "
            + " | ".join(
                _markdown(value)
                for value in (
                    item["path"],
                    item["asset_type"],
                    item["evidence_state"],
                    item["reuse"],
                    item["notes"],
                    _target_phase(str(item["asset_type"])),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _prepare_report_destinations(repository: Path) -> dict[str, Path]:
    report_dir = repository / "reports"
    try:
        report_stat = _lstat(report_dir)
    except FileNotFoundError:
        try:
            report_dir.mkdir()
        except FileExistsError:
            pass
        report_stat = _lstat(report_dir)
    except OSError as exc:
        raise _security_error(report_dir, f"cannot inspect reports directory: {exc}") from exc
    if _is_reparse_stat(report_stat):
        raise _security_error(report_dir, "reports is a symlink, junction, or reparse point")
    if not stat.S_ISDIR(report_stat.st_mode):
        raise _security_error(report_dir, "reports exists but is not a directory")
    try:
        resolved_reports = report_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _security_error(report_dir, f"cannot resolve reports directory: {exc}") from exc
    if resolved_reports.parent != repository or not _is_within(resolved_reports, repository):
        raise _security_error(report_dir, "reports does not resolve directly beneath repository root")

    reports = {
        "asset_inventory": report_dir / "asset_inventory.md",
        "reviewer_issue_matrix": report_dir / "reviewer_issue_matrix.md",
        "v2_reuse_map": report_dir / "v2_reuse_map.md",
        "v2_progress": report_dir / "v2_progress.md",
    }
    for destination in reports.values():
        _validate_report_destination(destination, repository, report_dir)
    return reports


def _validate_report_destination(destination: Path, repository: Path, report_dir: Path) -> None:
    try:
        report_stat = _lstat(report_dir)
    except OSError as exc:
        raise _security_error(report_dir, f"reports directory became unavailable: {exc}") from exc
    if _is_reparse_stat(report_stat) or not stat.S_ISDIR(report_stat.st_mode):
        raise _security_error(report_dir, "reports directory became a reparse point or non-directory")
    if report_dir.resolve(strict=True) != repository / "reports":
        raise _security_error(report_dir, "reports directory identity escaped repository")
    if destination.parent != report_dir or not _is_within(destination.parent, repository):
        raise _security_error(destination, "report destination is outside the contained reports directory")
    try:
        destination_stat = _lstat(destination)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _security_error(destination, f"cannot inspect report destination: {exc}") from exc
    if _is_reparse_stat(destination_stat):
        raise _security_error(destination, "report destination is a symlink or reparse point")
    if not stat.S_ISREG(destination_stat.st_mode):
        raise _security_error(destination, "report destination exists but is not a regular file")


def _root_identity(repository: Path) -> str:
    root_stat = _lstat(repository)
    identity = f"{repository}|{root_stat.st_dev}|{root_stat.st_ino}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _audit_metadata(
    repository: Path, assets: list[dict[str, object]], issues: list[dict[str, str]]
) -> tuple[str, str, str, str]:
    canonical_inventory = json.dumps(
        assets, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    inventory_digest = hashlib.sha256(canonical_inventory).hexdigest()
    root_identity = _root_identity(repository)
    completeness = "complete" if not issues else "incomplete"
    issue_digest = hashlib.sha256(
        json.dumps(issues, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    audit_id = hashlib.sha256(
        f"{SCANNER_VERSION}|{root_identity}|{inventory_digest}|{issue_digest}".encode("utf-8")
    ).hexdigest()[:24]
    metadata = "\n".join(
        [
            f"Audit ID: `{audit_id}`",
            f"Inventory digest: `{inventory_digest}`",
            f"Root identity: `{root_identity}`",
            f"Scanner version: `{SCANNER_VERSION}`",
            f"Completeness: `{completeness}`",
            f"Scan issues: `{len(issues)}`",
        ]
    )
    return audit_id, inventory_digest, root_identity, metadata


def _render_progress(
    report_dir: Path,
    metadata: str,
    *,
    completed: str,
    failed: str,
    unexpected: str,
    interpretation: str,
    next_gate: str,
) -> str:
    token = token_hex(8)
    render_path = report_dir / f".v2_progress.render.{token}.md"
    render_lock = render_path.with_name(f".{render_path.name}.lock")
    try:
        update_phase_progress(
            "Phase 0",
            completed=completed,
            failed=failed,
            unexpected=unexpected,
            interpretation=interpretation,
            next_gate=next_gate,
            path=render_path,
        )
        progress = render_path.read_text(encoding="utf-8")
    finally:
        if render_path.exists():
            render_path.unlink()
        if render_lock.exists():
            render_lock.unlink()
    body = progress.removeprefix("# V2 Progress\n").lstrip("\n")
    return "# V2 Progress\n\n" + metadata.rstrip() + "\n\n" + body


def _stage_text(destination: Path, content: str, label: str) -> Path:
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.{label}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _stage_backup(destination: Path) -> Path:
    with destination.open("rb") as source, NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _publish_report_set(
    repository: Path,
    rendered: dict[Path, str],
    replace_file: Callable[[Path, Path], object] | None = None,
) -> None:
    report_dir = repository / "reports"
    replace = replace_file or (lambda source, destination: source.replace(destination))
    stages: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    primary: BaseException | None = None
    with _cross_process_lock(report_dir / "v2_progress.md"):
        try:
            for destination, content in rendered.items():
                _validate_report_destination(destination, repository, report_dir)
                stages[destination] = _stage_text(destination, content, "stage")
            for destination in rendered:
                _validate_report_destination(destination, repository, report_dir)
                backups[destination] = _stage_backup(destination) if destination.exists() else None
            for destination in rendered:
                _validate_report_destination(destination, repository, report_dir)
                replace(stages[destination], destination)
                replaced.append(destination)
        except BaseException as exc:
            primary = exc
            for destination in reversed(replaced):
                try:
                    backup = backups[destination]
                    if backup is None:
                        if destination.exists():
                            destination.unlink()
                    else:
                        backup.replace(destination)
                except BaseException as rollback_error:
                    primary.add_note(
                        f"rollback failed for {destination}: {rollback_error.__class__.__name__}: {rollback_error}"
                    )
            raise
        finally:
            for temporary in [*stages.values(), *(p for p in backups.values() if p is not None)]:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError as cleanup_error:
                        if primary is not None:
                            primary.add_note(f"temporary cleanup failed for {temporary}: {cleanup_error}")


def run_audit(
    root: str | os.PathLike[str] = ".",
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
    *,
    open_file: Callable[[Path], BinaryIO] | None = None,
    read_chunk: Callable[[BinaryIO, int], bytes] | None = None,
    event_hook: Callable[[str, Path, int], None] | None = None,
    replace_file: Callable[[Path, Path], object] | None = None,
    expected_protocol_hash: str | None = None,
) -> dict[str, object]:
    repository = Path(root).resolve(strict=True)
    reports = _prepare_report_destinations(repository)
    try:
        scan = scan_repository(
            repository,
            max_hash_bytes=max_hash_bytes,
            open_file=open_file,
            read_chunk=read_chunk,
            event_hook=event_hook,
            expected_protocol_hash=expected_protocol_hash,
        )
    except AuditSecurityError:
        raise
    except BaseException as primary:
        try:
            _render_progress(
                repository / "reports",
                "",
                completed="No audit outputs were completed.",
                failed=f"Phase 0 scan failed: {primary.__class__.__name__}: {primary}",
                unexpected="No complete inventory is available.",
                interpretation="No scientific interpretation is permitted from an incomplete audit.",
                next_gate="Blocked until Phase 0 completes.",
            )
        except BaseException as progress_error:
            primary.add_note(
                f"progress rendering also failed: {progress_error.__class__.__name__}: {progress_error}"
            )
        raise

    assets = scan.assets
    issues = scan.issues
    audit_id, inventory_digest, root_identity, metadata = _audit_metadata(
        repository, assets, issues
    )
    skipped = sum(item["sha256"] == "not_computed:size_limit" for item in assets)
    if scan.complete:
        completed = (
            f"Inventoried {len(assets)} read-only historical assets and prepared one coherent report set. "
            "No training or inference was performed."
        )
        failed = "None."
        unexpected = (
            f"{skipped} oversized asset(s) were recorded with sha256=not_computed:size_limit; "
            "no source asset was modified."
        )
        next_gate = "Phase 1 fixed-split historical reproduction gate (0.5 percentage-point tolerance)."
    else:
        completed = (
            f"Inventoried {len(assets)} candidate assets, but the audit is incomplete. "
            "No training or inference was performed."
        )
        failed = f"{len(issues)} scan issue(s) require resolution or explicit --allow-incomplete acceptance."
        unexpected = "Incomplete read evidence is retained with explicit markers; no inconsistent hash is trusted."
        next_gate = (
            "Blocked until scan issues are resolved. Diagnostic reports are always generated; "
            "--allow-incomplete permits exit code 0 despite incompleteness."
        )
    interpretation = (
        "Phase 0 establishes provenance, validation state, and missing-evidence boundaries only; it does "
        "not validate universal scientific claims or mark missing experiments complete."
    )
    progress = _render_progress(
        repository / "reports",
        metadata,
        completed=completed,
        failed=failed,
        unexpected=unexpected,
        interpretation=interpretation,
        next_gate=next_gate,
    )
    rendered = {
        reports["asset_inventory"]: _inventory_report(
            assets, max_hash_bytes, metadata=metadata, issues=issues
        ),
        reports["reviewer_issue_matrix"]: _reviewer_report(assets, metadata=metadata),
        reports["v2_reuse_map"]: _reuse_report(assets, metadata=metadata),
        reports["v2_progress"]: progress,
    }
    _publish_report_set(repository, rendered, replace_file=replace_file)
    return {
        "asset_count": len(assets),
        "assets": assets,
        "issues": issues,
        "issue_count": len(issues),
        "complete": scan.complete,
        "audit_id": audit_id,
        "inventory_digest": inventory_digest,
        "root_identity": root_identity,
        "scanner_version": SCANNER_VERSION,
        "reports": {name: str(path) for name, path in reports.items()},
    }


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only AMC V2 Phase 0 asset audit.")
    parser.add_argument("--root", default=".", help="AWN repository root")
    parser.add_argument(
        "--max-hash-bytes",
        type=_non_negative_integer,
        default=DEFAULT_MAX_HASH_BYTES,
        help="maximum file size to hash (default: 128 MiB)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Diagnostic reports are always generated; this flag permits exit code 0 despite "
            "incompleteness"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.root, max_hash_bytes=args.max_hash_bytes)
    except AuditSecurityError as exc:
        print(f"Phase 0 security failure: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Phase 0 audit failure: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    status = "complete" if result["complete"] else "incomplete"
    print(
        f"Audited {result['asset_count']} assets ({status}; issues={result['issue_count']}); "
        "Phase 0 performed no training or inference."
    )
    print(f"audit_id: {result['audit_id']}")
    print(f"inventory_digest: {result['inventory_digest']}")
    for name, path in result["reports"].items():
        print(f"{name}: {path}")
    return 0 if result["complete"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
