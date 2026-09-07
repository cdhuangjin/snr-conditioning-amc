from __future__ import annotations

import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from .manifest import _cross_process_lock


def _write_atomic(path: Path, report: str) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        temporary_path.write_text(report, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def update_phase_progress(
    phase: str,
    completed: str,
    failed: str,
    unexpected: str,
    interpretation: str,
    next_gate: str,
    path: str | Path = "reports/v2_progress.md",
) -> None:
    """Create or replace one delimited phase section in the V2 progress report."""
    report_path = Path(path)
    start = f"<!-- V2_PHASE:{phase}:START -->"
    end = f"<!-- V2_PHASE:{phase}:END -->"
    section = (
        f"{start}\n"
        f"## {phase}\n\n"
        f"**Completed**\n\n{completed}\n\n"
        f"**Failed**\n\n{failed}\n\n"
        f"**Unexpected**\n\n{unexpected}\n\n"
        f"**Interpretation**\n\n{interpretation}\n\n"
        f"**Next gate**\n\n{next_gate}\n"
        f"{end}"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with _cross_process_lock(report_path):
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
        else:
            report = "# V2 Progress\n"

        pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
        if pattern.search(report):
            report = pattern.sub(lambda _: section, report, count=1)
        else:
            report = f"{report.rstrip()}\n\n{section}\n"

        _write_atomic(report_path, report)
