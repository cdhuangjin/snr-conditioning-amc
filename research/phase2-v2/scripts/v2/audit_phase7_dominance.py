"""Independent low-SNR dominant-class audit; no model fitting or source mutation."""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v2.phase7.diagnostics import verify_manifest,sha256


def pool_counts(rows):
    if not rows:raise ValueError('no count rows')
    totals=[];truth=[];n=0
    for r in rows:
        p=np.asarray(r['predicted_count']);t=np.asarray(r['true_count'])
        if p.ndim!=1 or t.shape!=p.shape or p.dtype.kind not in 'iu' or t.dtype.kind not in 'iu' or np.any(p<0) or np.any(t<0) or p.sum()!=r['n'] or t.sum()!=r['n'] or r['n']<=0:
            raise ValueError('count row does not match n')
        totals.append(p);truth.append(t);n+=int(r['n'])
    counts=np.sum(totals,axis=0);true_counts=np.sum(truth,axis=0);dominant=int(counts.argmax())
    return dict(n=n,predicted_count=counts.tolist(),true_count=true_counts.tolist(),true_frequency=(true_counts/n).tolist(),
        dominant_id=dominant,dominant_share=float(counts[dominant]/n))


def class_mapping(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    values=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='class_to_index' for t in n.targets)]
    if len(values)!=1:raise ValueError('ambiguous frozen class mapping')
    return {(k.decode('ascii') if isinstance(k,bytes) else k):v for k,v in values[0].items()}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def run(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('immutable review output exists')
    original=verify_manifest(source)
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    mapping_path=ROOT/'scripts/v2/phase1_reproduce.py';mapping=class_mapping(mapping_path)
    names=[name for name,index in sorted(mapping.items(),key=lambda p:p[1])]
    if names!=summary['class_names']:raise ValueError('diagnostic class names differ from frozen loader')
    am=mapping['AM-SSB'];wf=mapping['WBFM'];classes=len(names)
    if (am,wf)!=(10,3):raise ValueError('unexpected frozen mapping; investigate rather than relabel')
    with np.load(source/'raw_statistics.npz',allow_pickle=False) as raw:
        ids=raw['sample_ids'];labels=raw['y_true'];snr=raw['snr_db']
    low=snr<=-8
    expected_snr=list(range(-20,-7,2))
    if sorted(np.unique(snr[low]).tolist())!=expected_snr:raise ValueError('low-SNR bin scope differs')
    paths=[Path(path) for path in original['sources'] if Path(path).name=='predictions.npz' and Path(path).parent.parent==(ROOT/'results/v2/phase3').resolve()]
    records=[];per_snr=[];confusions=[];predictions=[]
    for path in sorted(paths):
        with np.load(path,allow_pickle=False) as b:
            model=str(b['model_id'].item());seed=int(b['seed']);pred=b['y_pred']
            if not np.array_equal(b['sample_ids'],ids) or not np.array_equal(b['y_true'],labels) or not np.array_equal(b['snr_db'],snr):raise ValueError('fixed raw/prediction row mismatch')
            if not np.array_equal(b['logits'].argmax(1),pred):raise ValueError('persisted logits and predictions differ')
        rows=[r for r in summary['predictions'] if r['model_id']==model and r['seed']==seed and r['snr_db']<=-8]
        if sorted(r['snr_db'] for r in rows)!=expected_snr:raise ValueError('summary low-SNR scope differs')
        pooled=pool_counts(rows)
        actual=np.bincount(pred[low],minlength=classes)
        truth=np.bincount(labels[low],minlength=classes)
        if not np.array_equal(actual,pooled['predicted_count']) or not np.array_equal(truth,pooled['true_count']):raise ValueError('summary/direct count mismatch')
        matrix=np.zeros((classes,classes),dtype=np.int64);np.add.at(matrix,(labels[low],pred[low]),1)
        errors=int(matrix.sum()-np.trace(matrix))
        record=dict(model_id=model,seed=seed,**pooled,dominant_class=names[pooled['dominant_id']],
            amssb_predicted_count=int(actual[am]),amssb_predicted_share=float(actual[am]/pooled['n']),
            wbfm_predicted_count=int(actual[wf]),wbfm_predicted_share=float(actual[wf]/pooled['n']),
            total_errors=errors,error_to_amssb_share=float((actual[am]-matrix[am,am])/errors) if errors else None,
            error_to_wbfm_share=float((actual[wf]-matrix[wf,wf])/errors) if errors else None,
            amssb_recall=float(matrix[am,am]/truth[am]),wbfm_recall=float(matrix[wf,wf]/truth[wf]),
            source_prediction=str(path),source_prediction_sha256=sha256(path),summary_direct_counts_identical=True)
        records.append(record);confusions.append(matrix);predictions.append(pred[low])
        for r in sorted(rows,key=lambda r:r['snr_db']):
            count=np.asarray(r['predicted_count']);mask=snr==r['snr_db']
            if not np.array_equal(count,np.bincount(pred[mask],minlength=classes)):raise ValueError('per-SNR count mismatch')
            index=int(count.argmax())
            per_snr.append(dict(model_id=model,seed=seed,snr_db=r['snr_db'],n=r['n'],dominant_id=index,dominant_class=names[index],dominant_share=float(count[index]/r['n']),amssb_share=float(count[am]/r['n']),wbfm_share=float(count[wf]/r['n'])))
    pairs={(r['model_id'],r['seed']) for r in records}
    if len(records)!=40 or pairs!={(f'M{i}',s) for i in range(8) for s in range(2022,2027)}:raise ValueError('expected all 40 original model/seed bundles')
    m0=[r for r in records if r['model_id']=='M0'];m0bins=[r for r in per_snr if r['model_id']=='M0']
    sources=dict(original['sources'])
    for path in [source/'manifest.json',*[source/name for name in original['artifacts']],Path(__file__),mapping_path,ROOT/'tests/v2/test_phase7_dominance_audit.py']:
        sources[str(path.resolve())]=sha256(path)
    result=dict(status='INDEPENDENT_DESCRIPTIVE_REVIEW_COMPLETE',not_phase7_completion=True,source_diagnostic_status=original['status'],class_mapping=mapping,
        source_manifest_sha256=sha256(source/'manifest.json'),source_summary_sha256=sha256(source/'summary.json'),split_hash=summary['split_hash'],
        low_snr_definition='snr_db <= -8; pooled original counts over seven registered bins',
        records=records,per_snr=per_snr,m0_amssb_dominant_seed_count=sum(r['dominant_class']=='AM-SSB' for r in m0),
        m0_amssb_dominant_bin_seed_count=sum(r['dominant_class']=='AM-SSB' for r in m0bins),
        inference_scope='descriptive association in fixed RML2016.10a split; no physical causal mechanism or universal dominant class')
    output.mkdir(parents=True)
    dump(output/'summary.json',result)
    fields=['model_id','seed','n','dominant_class','dominant_share','amssb_predicted_count','amssb_predicted_share','wbfm_predicted_count','wbfm_predicted_share','total_errors','error_to_amssb_share','error_to_wbfm_share','amssb_recall','wbfm_recall','source_prediction_sha256']
    with (output/'pooled_low_snr.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({key:r[key] for key in fields} for r in records)
    with (output/'per_snr.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per_snr[0]));w.writeheader();w.writerows(per_snr)
    np.savez_compressed(output/'recomputed_counts.npz',models=np.array([r['model_id'] for r in records]),seeds=np.array([r['seed'] for r in records]),
        predicted_counts=np.array([r['predicted_count'] for r in records]),true_counts=np.array([r['true_count'] for r in records]),confusion_matrices=np.stack(confusions),
        low_sample_ids=ids[low],low_y_true=labels[low],low_snr_db=snr[low],low_y_pred=np.stack(predictions),class_names=np.array(names))
    lines=['# Independent Phase 7 dominant-class review','',
        f"Verified original diagnostic manifest {sha256(source/'manifest.json')} and summary {sha256(source/'summary.json')}. All 40 model/seed bundles reproduce the original per-SNR and pooled counts exactly. Frozen loader mapping is AM-SSB=10 and WBFM=3; no current label-name mismatch was found.",'',
        '## M0 low-SNR result','',
        '| Seed | n | Dominant class | AM-SSB prediction count/share | WBFM prediction count/share |','|---|---:|---|---:|---:|']
    for r in m0:lines.append(f"| {r['seed']} | {r['n']} | {r['dominant_class']} | {r['amssb_predicted_count']} / {100*r['amssb_predicted_share']:.4f}% | {r['wbfm_predicted_count']} / {100*r['wbfm_predicted_share']:.4f}% |")
    lines+=['',f"AM-SSB dominates {result['m0_amssb_dominant_seed_count']}/5 pooled seeds and {result['m0_amssb_dominant_bin_seed_count']}/35 individual seed-by-low-SNR bins. Every true class contributes 1,400 of the 15,400 pooled test frames (9.0909%). The observed prediction concentration is not a test class-frequency imbalance.",'',
        'The share column divides all predictions to that class by all low-SNR frames. It is different from error-to-class share (class-directed errors divided by all errors) and from class recall. The JSON/CSV/NPZ preserve all three quantities and their underlying counts. Pooling sums counts; it does not average percentages from potentially unequal bins.','',
        '## Interpretation and scope','',
        'The original WBFM-attractor premise is contradicted for these M0 checkpoints and this fixed low-SNR protocol. The defensible observation is AM-SSB prediction concentration. This does not establish a physical reason, an inevitable collapse, or a universal AM-SSB attractor. It also does not retrospectively prove that another historical checkpoint/protocol had a label error.','',
        'Keep the registered WBFM-removal and RMS controls unchanged and retain their outcomes. WBFM removal now tests a specified non-dominant class-removal sensitivity and the historical hypothesis; it cannot by itself explain the actual dominant class. RMS remains a signal-statistics sensitivity. Ten-class removal tasks differ from the original eleven-class task; an accuracy increase alone is insufficient mechanistic evidence.','',
        'A separately versioned remove-AM-SSB five-seed follow-up is warranted if Phase 7 is to test the actual dominant-class mechanism. Selecting AM-SSB used the existing test predictions: freeze the new training protocol prospectively, but label the target selection post hoc/exploratory. Independent cross-dataset and synthetic evidence are needed for broader confirmation. No new training is launched by this audit.','',
        'Existing WBFM-centered feature distances cannot be renamed into AM-SSB evidence. Additional AM-SSB-centered feature and class-wise error-flow analyses should use the persisted real feature/prediction bundles. The Gaussian surrogate is a shape reference, not the recovered physical noise component.','',
        '## Other model scope (same seven-bin pool)','',
        '| Model | Dominant names across five seeds | Mean dominant share |','|---|---|---:|']
    for model in sorted({r['model_id'] for r in records}):
        rr=[r for r in records if r['model_id']==model]
        lines.append(f"| {model} | {', '.join(sorted({r['dominant_class'] for r in rr}))} | {100*np.mean([r['dominant_share'] for r in rr]):.4f}% |")
    lines+=['','All values are generated from the registered summary and independently checked against original persisted logits/predictions. This report completes a read-only interpretation audit, not Phase 7.']
    (output/'interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    dump(output/'manifest.json',dict(status='INDEPENDENT_DESCRIPTIVE_REVIEW_COMPLETE',sources=sources,artifacts=artifacts))
    print(json.dumps(dict(output=str(output),m0_amssb_dominant_seed_count=result['m0_amssb_dominant_seed_count'],m0_amssb_dominant_bin_seed_count=result['m0_amssb_dominant_bin_seed_count'],source_summary_sha256=result['source_summary_sha256']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
