"""Read-only provenance audit of the existing 2016.04C text mirror."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT.parents[1]
API = "https://api.github.com/repos/dannis999/RML2016.04C"


def get(suffix):
    request = urllib.request.Request(API + suffix, headers={"User-Agent": "Codex-research-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    output = ROOT / "reports/phase11_source_audit_20260906.json"
    if output.exists():
        raise FileExistsError(output)
    repo = get("")
    commit = get("/commits/" + repo["default_branch"])["sha"]
    tree = get("/git/trees/" + commit + "?recursive=1")
    if tree.get("truncated"):
        raise ValueError("remote tree is incomplete")
    local = PRIMARY / "downloads/RML2016.04C-main"
    rows = []
    for entry in tree["tree"]:
        if entry["type"] != "blob":
            continue
        path = (local / entry["path"]).resolve()
        if not path.is_relative_to(local.resolve()):
            raise ValueError("remote path escapes local mirror")
        data = path.read_bytes() if path.is_file() else None
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() if data is not None else None
        rows.append({"file": entry["path"], "remote_git_blob": entry["sha"], "local_git_blob": blob,
                     "local_sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
                     "bytes": len(data) if data is not None else None, "match": blob == entry["sha"]})
    remote_files = {row["file"] for row in rows}
    local_files = {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()}
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "repository": repo["html_url"],
              "commit": commit, "tree": tree["sha"], "files": rows,
              "matched": sum(row["match"] for row in rows), "total": len(rows),
              "extra_local_files": sorted(local_files - remote_files),
              "interpretation": "Content match to a public backup, not authentication against the original DeepSig distribution or recovery of the historical download time."}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("repository", "commit", "matched", "total", "extra_local_files")}))


if __name__ == "__main__":
    main()
