"""Verify distributed evidence bytes and paired manuscript effects without training."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import t

E = Path(__file__).resolve().parents[1] / "results/paper"
def read(name):
    return json.loads((E / name).read_text(encoding="utf-8"))

def main():
    manifest = read("release-manifest.json")
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((E / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("Evidence hash differs: " + name)
    count = 0
    for family in ["04c", "synthetic"]:
        rows = read(family + "_rows.json")
        for p in read(family + "_paired.json"):
            if p.get("group") != "overall" or p.get("metric") != "accuracy":
                continue
            if family == "synthetic" and p.get("kind") != "M6_minus_baseline":
                continue
            if family == "04c":
                a = {r["seed"]:r["metrics"]["overall"]["accuracy"] for r in rows if r["model"] == "M6" and r["condition"] == p["condition"]}
                b = {r["seed"]:r["metrics"]["overall"]["accuracy"] for r in rows if r["model"] == "M0"}
            else:
                a = {r["seed"]:r["metrics"]["overall"]["accuracy"] for r in rows if r["model"] == "M6" and r["channel"] == p["channel"] and r["condition"] == p["condition"]}
                b = {r["seed"]:r["metrics"]["overall"]["accuracy"] for r in rows if r["model"] == p["baseline"] and r["channel"] == p["channel"] and r["condition"] == p["condition"]}
            assert set(a) == set(b) == set(range(2022, 2027))
            d = np.array([a[s] - b[s] for s in sorted(a)])
            half = t.ppf(.975, 4) * d.std(ddof=1) / np.sqrt(5)
            assert abs(d.mean() - p["mean"]) < 1e-12
            assert np.allclose([d.mean()-half, d.mean()+half], p["ci95"], atol=1e-12, rtol=0)
            print(f"{family} {p.get('channel','')} {p['condition']}: {100*d.mean():+.6f} pp")
            count += 1
    print(f"PASS: {len(manifest['sha256'])} evidence hashes; {count} paired effects.")

if __name__ == "__main__":
    main()
