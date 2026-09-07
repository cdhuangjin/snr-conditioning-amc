"""Verify local 04C conversion and record a deterministic pre-split dedup map."""
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT.parents[1]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    output = ROOT / "results/v2/phase11_preflight/04c_source_dedup"
    if output.exists():
        raise FileExistsError(output)
    source = PRIMARY / "data/RML2016.04c.dat"
    source_sha = sha(source)
    mirror_root = PRIMARY / "downloads/RML2016.04C-main/2016.04C"
    mirror_evidence = ROOT / "reports/phase11_source_audit_20260906.json"
    mirror = json.loads(mirror_evidence.read_text(encoding="utf-8"))
    assert mirror["matched"] == mirror["total"] == 221 and not mirror["extra_local_files"]
    verified = {row["file"]: row["local_sha256"] for row in mirror["files"]}
    with source.open("rb") as handle:
        cells = pickle.load(handle, encoding="bytes")
    keys = sorted(cells)
    if len(keys) != 220:
        raise ValueError("expected all 220 class/SNR cells")
    groups = defaultdict(list)
    labels, snrs, within, counts = [], [], [], []
    source_names = sorted({key[0] for key in keys})
    cell_names = []
    offset = 0
    for class_name, snr in keys:
        values = np.asarray(cells[(class_name, snr)], dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (2, 128) or not np.isfinite(values).all():
            raise ValueError("invalid source cell")
        path = mirror_root / f"{class_name.decode()} {snr}.txt"
        if sha(path) != verified[path.relative_to(mirror_root.parent).as_posix()]:
            raise ValueError("mirror content changed since remote content audit")
        n = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                z = np.asarray([complex(token.strip("()")) for token in line.split()], dtype=np.complex64)
                reconstructed = np.stack([z.real, z.imag])
                if n >= len(values) or not np.array_equal(values[n], reconstructed):
                    raise ValueError(f"pickle/text conversion differs: {path.name}:{n}")
                # Canonicalize signed zero before numeric duplicate hashing.
                canonical = np.asarray(values[n], dtype="<f4").copy()
                canonical[canonical == 0] = 0.0
                digest = hashlib.sha256(canonical.tobytes()).hexdigest()
                groups[digest].append(offset+n)
                n += 1
        if n != len(values):
            raise ValueError("mirror/source cell row count differs")
        label = source_names.index(class_name)
        labels.extend([label]*n); snrs.extend([int(snr)]*n); within.extend(range(n))
        counts.append(n); cell_names.append(f"{class_name.decode()}:{snr}")
        offset += n
        print(f"verified {class_name.decode()} snr={snr}: {n} rows", flush=True)
    labels = np.asarray(labels); snrs = np.asarray(snrs); within = np.asarray(within)
    retained, duplicates, conflicts = [], [], []
    for indices in groups.values():
        identities = {(int(labels[i]), int(snrs[i])) for i in indices}
        if len(identities) > 1:
            conflicts.extend(indices)  # no arbitrary label/SNR assignment
        else:
            retained.append(indices[0]); duplicates.extend(indices[1:])
    retained, duplicates, conflicts = map(lambda x: np.asarray(sorted(x), dtype=np.int64), (retained, duplicates, conflicts))
    if len(retained)+len(duplicates)+len(conflicts) != offset:
        raise ValueError("dedup accounting does not close")
    if sha(source) != source_sha:
        raise ValueError("input changed during audit")
    output.mkdir(parents=True)
    np.savez_compressed(output/"dedup_map.npz", retained_indices=retained, duplicate_indices=duplicates,
                        conflicting_indices=conflicts, original_labels=labels, original_snrs=snrs,
                        original_within_cell=within, class_names=np.asarray([name.decode() for name in source_names]))
    report = {"status":"SOURCE_AND_DEDUP_AUDITED_NOT_TRAINED", "source":str(source), "source_sha256":source_sha,
              "mirror_repository":mirror["repository"], "mirror_commit":mirror["commit"],
              "verified_conversion_rows":offset, "cells":dict(zip(cell_names, counts)),
              "retained_rows":len(retained), "same_cell_duplicates_removed":len(duplicates),
              "conflicting_label_or_snr_rows_removed":len(conflicts),
              "dedup_rule":"SHA256 of little-endian float32 IQ after signed-zero canonicalization; retain first same-cell representative; drop all cross-label/SNR conflicts before any split",
              "scope":"Exact numeric duplicates only; near duplicates/channel-related samples require separate limitations. No split or training performed."}
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/"manifest.json").write_text(json.dumps({"status":report["status"], "files":files,
        "dependencies":{str(source):source_sha, str(mirror_evidence):sha(mirror_evidence), str(Path(__file__).resolve()):sha(__file__)}}
        ,indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","verified_conversion_rows","retained_rows","same_cell_duplicates_removed","conflicting_label_or_snr_rows_removed")}))


if __name__ == "__main__":
    main()
