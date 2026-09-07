"""Update human-facing evidence without modifying frozen source registries."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.phase3_analysis import _validate_existing
from v2.progress import update_phase_progress


def main():
    spec = importlib.util.spec_from_file_location("phase4_evidence", ROOT / "scripts/v2/run_phase4.py")
    phase4 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = phase4
    spec.loader.exec_module(phase4)
    p3 = ROOT / "results/v2/phase3_conditioning"
    manifest3 = json.loads((p3 / "manifest.json").read_text(encoding="utf-8"))
    _validate_existing(p3, manifest3["analysis_fingerprint"])
    p4, manifest4 = phase4._current_complete(ROOT / "results/v2/phase4_collapse_metrics")
    summary = json.loads((p3 / "conditioning_summary.json").read_text(encoding="utf-8"))
    summary4 = json.loads((p4 / "summary.json").read_text(encoding="utf-8"))
    rows = ["# Verified Phase 3–4 evidence", "", "These phases are complete; the full V2 plan is not complete.", "",
            "| Model | Accuracy mean ± SD | n |", "|---|---:|---:|"]
    for model, result in summary["models"].items():
        metric = result["overall_accuracy"]
        rows.append(f"| {model} | {100*metric['mean']:.4f} ± {100*metric['std']:.4f}% | {metric['n']} |")
    rows += ["", f"Phase 3: `{p3.relative_to(ROOT).as_posix()}`, {len(manifest3['files'])} hash-verified files.",
             f"Phase 4: `{p4.relative_to(ROOT).as_posix()}`, {len(manifest4['artifacts'])} hash-verified files, {len(manifest4['feature_bindings'])} checkpoint-derived feature bundles.",
             "", "## Reviewer evidence mapping", "",
             "- Conditioning capacity: five matched seeds, saved predictions and paired statistics; M3 is an additive hidden preactivation shift through a nonlinear classifier, not arbitrary independent per-bin boundaries.",
             "- Collapse diagnostics: 800 model×seed×SNR cells, continuous concentration/BA/entropy/HHI/Gini/ECE and threshold-sensitivity tables. This does not establish universal collapse or a phase transition.",
             "- Feature geometry: real M0/M3 and validation-selected stronger-conditioner checkpoint features, fixed test IDs, classifier replay and hash-bound manifests.",
             "- Interpolation: all discrete bins were observed during training; unseen-SNR interpolation remains unsupported.",
             "- Next gates: toy counterexamples, preprocessing/streaming audit, WBFM controls, estimator side-channel and robustness controls, cross-dataset and synthetic benchmarks.",
             "", "## Verification provenance", "",
             "`results/v2/analysis_readiness_20260906_stdout.log`: default real 40/40 checkpoint replay.",
             "`results/v2/frozen_source_audit_20260906.json`: 520/520 recorded source SHA256 comparisons.",
             "`results/v2/validation_allfixed_20260906.log`: fresh Phase 3/4 regression results.",
             "The Phase 4 Windows lock repair affects scheduling only; frozen training files/configs are unchanged."]
    (ROOT / "reports/v2_phase34_verified.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    update_phase_progress("Phase 3", "40/40 runs completed (10 reused, 30 trained); real checkpoint replay, independent frozen-source hashes, and 37-file formal analysis verified.",
                          "None outstanding; historical interrupted attempts preserved.",
                          "M7 capacity control is not the most accurate model; keep this valid result. Unseen-SNR interpolation is unsupported.",
                          "Matched-seed, setting-specific results only. Formal tables/figures: results/v2/phase3_conditioning; evidence map: reports/v2_phase34_verified.md.",
                          "Phase 4 verified; continue Phase 5–12 under the original V2 plan.", path=ROOT / "reports/v2_progress.md")
    print(json.dumps({"phase3_files":len(manifest3["files"]), "phase4_files":len(manifest4["artifacts"]), "phase4_status":summary4["status"]}))


if __name__ == "__main__":
    main()
