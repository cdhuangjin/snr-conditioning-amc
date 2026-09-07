"""Give every original numbered reviewer comment a stable source-linked ID."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/v2/final_preflight/reviewer_sources'
OUTPUT = ROOT/'results/v2/final_preflight/reviewer_issue_catalog'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pages = json.loads((SOURCE/'pages.json').read_text(encoding='utf-8'))
    selected = [p for p in pages if p['document'] == 'reviews']
    chunks, intervals = [], []
    offset = 0
    for page in selected:
        text = page['text']+'\n\n'
        chunks.append(text)
        intervals.append((offset, offset+len(text), page['page']))
        offset += len(text)
    whole = ''.join(chunks)
    headers = list(re.finditer(r'Reviewer ([134]) - Initial submission[^\n]*', whole))
    if [int(m[1]) for m in headers] != [1, 3, 4]:
        raise ValueError('reviewer headings incomplete')
    issues = []
    for index, header in enumerate(headers):
        reviewer = int(header[1])
        end = headers[index+1].start() if index+1 < len(headers) else len(whole)
        text = whole[header.end():end]
        numbers = list(re.finditer(r'(?m)^(\d{1,2})\.\s+', text))
        expected = {1:14, 3:5, 4:6}[reviewer]
        if [int(m[1]) for m in numbers] != list(range(1, expected+1)):
            raise ValueError('numbered source comments incomplete')
        for j, number in enumerate(numbers):
            stop = numbers[j+1].start() if j+1 < len(numbers) else len(text)
            quote = text[number.end():stop]
            quote = re.split(r'\nReviewer [134]\s*\nComments|\nAttached file:', quote)[0].strip()
            start_absolute, end_absolute = header.end()+number.start(), header.end()+stop
            covered = [page for begin, finish, page in intervals if begin < end_absolute and finish > start_absolute]
            issues.append(dict(issue_id=f'R{reviewer}.{int(number[1]):02d}', reviewer=reviewer, number=int(number[1]),
                               source_pdf='reviews.pdf', source_pdf_sha256=sha(SOURCE/'reviews.pdf'),
                               source_pages=covered, original_comment=quote,
                               response_status='NOT_FINAL_AUDITED', evidence=[], manuscript_action=None,
                               reviewer_response_action=None))
    OUTPUT.mkdir(parents=True)
    value = dict(status='SOURCE_CATALOG_NOT_RESOLVED_EVIDENCE', count=len(issues), issues=issues,
                 required_final_fields=['issue_id','code_path','run_ids','datasets','seeds','split_hash','protocol_hash',
                                        'json_npz','figure_table','statistical_summary','manuscript_action','reviewer_response_action'],
                 note='All 25 numbered comments retained. Reviewer4 attachment is preserved separately; minor sub-bullets remain within R4.06.')
    (OUTPUT/'issues.json').write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(status=value['status'], files={'issues.json':sha(OUTPUT/'issues.json')},
         dependencies={str(p):sha(p) for p in [SOURCE/'manifest.json', SOURCE/'pages.json', SOURCE/'reviews.pdf', Path(__file__).resolve()]}), indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':value['status'],'count':len(issues),'reviewers':{r:sum(i['reviewer']==r for i in issues) for r in (1,3,4)}}))


if __name__ == '__main__':
    main()
