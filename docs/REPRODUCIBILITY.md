# Reproducibility status

## Verified for this release

- The retained source was copied without rewriting model or numerical experiment logic.
- 118 targeted tests passed (conditioning, metrics, statistics, independent waveform core).
- The independent-waveform CPU smoke completed all three models, two epochs each, with estimator fitting, pilot evaluation, and output validation.
- `scripts/verify_paper_results.py` checks release evidence hashes and recomputes 04C/independent paired mean effects and t intervals from five per-seed rows.
- A separate fresh checkout is used to check the documented quick start with the existing environment. No new formal five-seed 100-epoch training is claimed in release preparation.

## Full historical training

From `research/phase2-v2`, entry points accept their registered configurations. Examples:

```sh
python scripts/v2/run_phase3.py --config configs/v2/phase3.yaml
python scripts/v2/run_phase5.py --config configs/v2/phase5.yaml
python scripts/v2/run_phase8.py --config configs/v2/phase8.yaml
python scripts/v2/run_phase9.py --config configs/v2/phase9.yaml
python scripts/v2/run_phase10.py --config configs/v2/phase10.yaml
python scripts/v2/prepare_04c_split.py
python scripts/v2/run_phase11.py --config configs/v2/phase11_04c.yaml
python scripts/v2/run_phase12.py --config configs/v2/phase12_execution.yaml
```

These are original entry points, **not a claim that a fresh clone alone can execute the entire sequence**. Later phases require prior completed training, immutable checkpoints, estimator artifacts, and provenance receipts. The original workflow also includes preprocessing and primary-evidence receipt checks in phases 1/2/6/7. This release includes their implementation and configs, but not the full multi-gigabyte checkpoint/receipt history. Thus, summary reanalysis and smoke replay are directly supported; full historical cached-run replay is not yet packaged as a single command.

The formal settings use 100 epochs, five seeds, Adam with learning rate 0.001 and batch size 128. Independent phase12 has 15 formal training runs. CPU smoke overrides data size, epochs, seeds, and device and is always marked non-scientific. The exact GPU/runtime metadata, where retained, is in the released result/protocol JSON files. Resource demand for a new full run has not been benchmarked in a clean release environment.

## Provenance and release paths

Historical `*_manifest.json`, `*_current.json`, and `source_audit.json` records describe the original workspace, not a newly validated portable receipt chain. One path-only normalization was made to `noise_rows.json`: the former local user-directory prefix was removed from artifact references; numerical values are unchanged. `results/paper/release-manifest.json` hashes the distributed files, superseding original source hashes only for checking the bytes of this release. Editorial correspondence and previous manuscript text were deliberately excluded from the research dataset.
