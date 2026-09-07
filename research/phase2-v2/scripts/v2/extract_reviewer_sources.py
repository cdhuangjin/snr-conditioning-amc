"""Preserve exact supplied review PDFs and page-addressable extracted text."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parents[2]/'PHYCOM-D-26-02127-reviews.zip'
OUTPUT = ROOT/'results/v2/final_preflight/reviewer_sources'
ENTRIES = {'PHYCOM-D-26-02127-reviews.pdf': 'reviews',
           'Initial submission/Reviewer 4/Comments to the Author.pdf': 'reviewer4_attachment'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pages = []
    with zipfile.ZipFile(SOURCE) as archive:
        for member, name in ENTRIES.items():
            raw = archive.read(member)
            (OUTPUT/(name+'.pdf')).write_bytes(raw)
            reader = PdfReader(io.BytesIO(raw))
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if not text.strip():
                    raise ValueError('empty extracted page requires visual/OCR inspection')
                pages.append(dict(document=name, zip_member=member, pdf_sha256=hashlib.sha256(raw).hexdigest(), page=number, text=text))
            (OUTPUT/(name+'.txt')).write_text('\n\n'.join(f'PAGE {p["page"]}\n{p["text"]}' for p in pages if p['document']==name), encoding='utf-8')
    (OUTPUT/'pages.json').write_text(json.dumps(pages, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(status='EXTRACTED_SOURCE_NOT_ISSUE_CLOSURE',
        files={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        dependencies={str(SOURCE):sha(SOURCE), str(Path(__file__).resolve()):sha(__file__)}), indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'documents': len(ENTRIES), 'pages': len(pages), 'status': 'EXTRACTED_SOURCE_NOT_ISSUE_CLOSURE'}))


if __name__ == '__main__':
    main()
