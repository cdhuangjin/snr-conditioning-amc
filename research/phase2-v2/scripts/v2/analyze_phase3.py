#!/usr/bin/env python3
"""Publish Phase 3 conditioning analysis from registered artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.phase3_analysis import AnalysisConflictError, AnalysisNotReadyError, analyze_phase3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--source-manifest")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = analyze_phase3(
            repository_root=arguments.repository_root,
            source_manifest=arguments.source_manifest,
            dry_run=arguments.dry_run,
        )
    except AnalysisNotReadyError as exc:
        print(json.dumps({"ready": False, "error": str(exc)}))
        return 2
    except (AnalysisConflictError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}))
        return 3
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("ready", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
