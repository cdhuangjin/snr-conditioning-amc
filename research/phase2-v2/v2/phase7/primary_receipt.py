"""Scoped primary acceptance; raw Phase 7 manifests remain immutable."""
import json
from pathlib import Path
from .controls import CONTROLS, SEEDS, atomic_json, file_hash, verify_control_manifest
from v2.phase6.composite import manifest_closure

KIND='phase7_primary_scoped_receipt'
PENDING=['remove_amssb_five_seed_posthoc','cross_dataset_channel_phase11']


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root):raise ValueError('unsafe receipt path')
    return p


def relative(root,path):
    p=Path(path).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('external receipt source')
    return p.relative_to(Path(root).resolve()).as_posix()


def snapshot_sources(root,destination,sources):
    """Snapshot every manifest basename, avoiding legacy recursive schema ambiguity."""
    destination=Path(destination);dependencies={};index=[]
    for locator,expected in sorted(sources.items()):
        path=Path(locator).resolve();name=relative(root,path)
        if file_hash(path)!=expected:raise ValueError('source hash mismatch')
        if path.name=='manifest.json':
            snapshot=f'snapshots/{len(index):04d}_{expected[:16]}.snapshot.json'
            target=destination/snapshot;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
            index.append(dict(original_path=name,sha256=expected,snapshot_path=snapshot))
        else:dependencies[name]=expected
    atomic_json(destination/'snapshot_index.json',index)
    return dependencies


def validate_snapshots(root,directory):
    seen=set()
    for row in load(Path(directory)/'snapshot_index.json'):
        original=safe(root,row['original_path']);snapshot=safe(directory,row['snapshot_path'])
        if row['original_path'] in seen:raise ValueError('duplicate original snapshot')
        seen.add(row['original_path'])
        if original.name!='manifest.json' or not snapshot.name.endswith('.snapshot.json'):raise ValueError('invalid snapshot identity')
        if file_hash(original)!=row['sha256']:raise ValueError('original manifest locator changed')
        if file_hash(snapshot)!=row['sha256']:raise ValueError('snapshot hash mismatch')


def require_matrix(root,summary):
    rows=summary.get('runs',[])
    if len(rows)!=10 or {(r['control'],r['seed']) for r in rows}!={(c,s) for c in CONTROLS for s in SEEDS}:
        raise ValueError('exactly ten completed control runs required')
    families={c:set() for c in CONTROLS}
    for row in rows:
        directory=Path(row['path']).resolve();relative(root,directory)
        m=verify_control_manifest(directory)
        if (m['control'],m['seed'])!=(row['control'],row['seed']) or file_hash(directory/'manifest.json')!=row['manifest_sha256']:
            raise ValueError('matrix identity mismatch')
        if load(directory/'metrics.json')!=row['metrics']:raise ValueError('matrix metrics mismatch')
        history=load(directory/'epochs.json')
        if len(history)!=100 or [r['epoch'] for r in history]!=list(range(100)):raise ValueError('incomplete epoch history')
        families[m['control']].add(m['protocol_hash'])
    if any(len(v)!=1 for v in families.values()):raise ValueError('mixed control protocol families')
    return rows


def raw_sources(directory,expected_status):
    directory=Path(directory);m=load(directory/'manifest.json')
    if m['status']!=expected_status:raise ValueError('raw component status mismatch')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=set(m['artifacts']):raise ValueError('raw component closure mismatch')
    result={str(directory/'manifest.json'):file_hash(directory/'manifest.json'),**m['sources']}
    result.update({str(safe(directory,n)):h for n,h in m['artifacts'].items()})
    for name,h in result.items():
        if file_hash(name)!=h:raise ValueError('raw component hash mismatch')
    return result


def decision():
    return dict(kind=KIND,status='COMPLETE',allow_phase8=True,full_phase7_complete=False,
        completion_meaning='original_diagnostics_and_ten_prespecified_controls_only',pending=PENDING,
        dominant_premise='M0 low-SNR dominance is AM-SSB; WBFM removal is the prespecified control, not evidence of WBFM causation')


def write_component(root,directory,sources,payload):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    deps=snapshot_sources(root,directory,sources)
    atomic_json(directory/'decision.json',payload)
    atomic_json(directory/'manifest.json',dict(status='COMPLETE',files={p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*') if p.is_file()},dependencies=deps))


def validate_primary_gate(root,gate):
    root=Path(root).resolve()
    try:
        if not isinstance(gate,dict):gate=load(gate)
        if any(gate.get(k)!=v for k,v in decision().items()):raise ValueError('primary gate scope mismatch')
        directories={}
        for key in ('manifest','diagnostics_manifest'):
            path=safe(root,gate[key+'_path'])
            if path.name!='manifest.json' or file_hash(path)!=gate[key+'_sha256']:raise ValueError('primary gate manifest hash mismatch')
            if load(path).get('status')!='COMPLETE':raise ValueError('primary component incomplete')
            manifest_closure(root,path);validate_snapshots(root,path.parent);directories[key]=path.parent
        primary=directories['manifest'];payload=load(primary/'decision.json')
        if payload['decision']!=decision():raise ValueError('primary decision mismatch')
        summary_path=safe(root,payload['summary_path'])
        if load(primary/'manifest.json')['dependencies'].get(payload['summary_path'])!=file_hash(summary_path):raise ValueError('unbound matrix summary')
        rows=require_matrix(root,load(summary_path))
        originals={r['original_path']:r['sha256'] for r in load(primary/'snapshot_index.json')}
        for row in rows:
            if originals.get(relative(root,Path(row['path'])/'manifest.json'))!=row['manifest_sha256']:raise ValueError('unbound raw control manifest')
        if load(directories['diagnostics_manifest']/'decision.json')!={'completion_meaning':'diagnostics_only','status':'COMPLETE'}:raise ValueError('diagnostic scope mismatch')
        return primary
    except (OSError,KeyError,TypeError) as exc:raise ValueError(f'primary receipt malformed: {exc}') from exc


def publish_primary(root,summary_path,diagnostics,dominance_review,*,replay,extra_sources=()):
    """Replay callback must perform the frozen checkpoint/prediction verification."""
    from .controls import digest
    root=Path(root).resolve();summary_path=Path(summary_path).resolve();summary=load(summary_path)
    rows=require_matrix(root,summary)  # fail before any output if a run is missing
    sources={str(summary_path):file_hash(summary_path),str(Path(__file__).resolve()):file_hash(__file__)}
    sources.update({str(Path(p).resolve()):file_hash(p) for p in extra_sources})
    for row in rows:
        replay(row)
        sources.update(raw_sources(row['path'],'COMPLETE'))
    algebra=Path(summary['algebraic_path'])
    if file_hash(algebra/'manifest.json')!=summary['algebraic_manifest_sha256']:raise ValueError('algebraic identity mismatch')
    algebra_summary=load(algebra/'summary.json')
    if [r['seed'] for r in algebra_summary['baselines_and_remaps']]!=list(SEEDS):raise ValueError('five algebraic baselines required')
    sources.update(raw_sources(algebra,'COMPLETE'))
    review_sources=raw_sources(dominance_review,'INDEPENDENT_DESCRIPTIVE_REVIEW_COMPLETE')
    sources.update(review_sources)
    diag_sources=raw_sources(diagnostics,'DIAGNOSTICS_COMPLETE_CONTROLS_PENDING')
    # Conflicting hash claims are already rejected against actual bytes above.
    identity=digest(dict(sources=sources,diagnostics=diag_sources,decision=decision()))
    base=root/'results/v2/phase7_primary/receipts'/identity[:16]
    if base.exists():raise ValueError('immutable receipt exists; validate existing gate')
    write_component(root,base/'diagnostics',diag_sources,dict(completion_meaning='diagnostics_only',status='COMPLETE'))
    diagnostic_manifest=base/'diagnostics/manifest.json'
    # This one dependency is standard and may be recursively visited.
    write_component(root,base/'primary',sources,dict(decision=decision(),summary_path=relative(root,summary_path)))
    primary_manifest=base/'primary/manifest.json';m=load(primary_manifest)
    m['dependencies'][relative(root,diagnostic_manifest)]=file_hash(diagnostic_manifest);atomic_json(primary_manifest,m)
    gate=dict(**decision(),schema_version=1,manifest_path=relative(root,primary_manifest),manifest_sha256=file_hash(primary_manifest),
        diagnostics_manifest_path=relative(root,diagnostic_manifest),diagnostics_manifest_sha256=file_hash(diagnostic_manifest))
    validate_primary_gate(root,gate)
    gate_path=root/'results/v2/phase7_controls/gate.json'
    if gate_path.exists():raise ValueError('existing gate must not be overwritten')
    gate_path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(gate_path,gate)
    return gate
