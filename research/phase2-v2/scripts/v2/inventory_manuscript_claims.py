"""Read-only required-term inventory; it is not a scientific claim clearance."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[2]
PATTERNS = {
    'inevitable': r'\binevitabl[ey]\b',
    'necessary': r'\bnecessary\b',
    'sufficient': r'\bsufficient\b',
    'cannot_or_impossible': r'\b(?:cannot|impossible)\b',
    'universal': r'\buniversally?\b|\buniversal\b',
    'phase_transition': r'\bphase[\s-]+transition\b',
    'arbitrary_per_snr_boundary': r'\barbitrary[^\n.]{0,80}(?:boundar|per[\s-]*SNR)',
    'continuous_embedding': r'\bcontinuous[\s-]+embedding\b',
    'effective_feature_mass': r'\beffective[\s-]+feature[\s-]+mass\b',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sources = [PROJECT/'TCCN_manuscript.md', PROJECT/'TCCN_submission/main.tex']
    output = ROOT/'results/v2/final_preflight/claim_inventory_20260906'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for path in sources:
        value = path.read_text(encoding='utf-8')
        for line_number, line in enumerate(value.splitlines(), 1):
            for term, pattern in PATTERNS.items():
                for match in re.finditer(pattern, line, flags=re.I):
                    rows.append(dict(source=str(path), source_sha256=digest(path), line=line_number,
                                     term=term, matched_text=match.group(), context=line.strip(),
                                     review_status='REQUIRES_CONTEXTUAL_REVIEW'))
    output.mkdir(parents=True)
    value = dict(status='UNREVIEWED_PRE_REVISION_INVENTORY', matches=rows,
                 limitations=['A lexical hit may be a qualified or negated claim; no automatic rewriting or clearance.',
                              'These are existing drafts, not the future V2 manuscript. Rescan final deliverables after all results.'],
                 patterns=PATTERNS)
    (output/'inventory.json').write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(dict(status=value['status'], files={'inventory.json':digest(output/'inventory.json')},
            dependencies={str(p):digest(p) for p in [*sources, Path(__file__).resolve()]}), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=value['status'], sources=len(sources), matches=len(rows))))


if __name__ == '__main__':
    main()
