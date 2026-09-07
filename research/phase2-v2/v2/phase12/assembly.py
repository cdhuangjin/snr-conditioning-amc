"""Bounded per-frame assembly; observation identities distinguish paired channels."""
from pathlib import Path
import json
import numpy as np
from v2.phase8.evidence import sha, dump
from .core import FrameID, COUNTS, PURPOSES, FIR, generate_frame

ROOT=Path(__file__).resolve().parents[2]
ARRAYS={'signals':('nct',np.float32,(2,128)), 'clean':('clean',np.complex128,(128,)),
        'observed':('payload',np.complex128,(128,)), 'pilot':('pilot',np.complex128,(64,)),
        'payload_noise':('payload_noise',np.complex128,(128,)), 'pilot_noise':('pilot_noise',np.complex128,(64,))}
SCALARS=('target_snr_db','cfo','gain_db','gain_amplitude','clean_power','post_channel_power',
         'noise_variance_before_gain','noise_variance_after_gain','received_expected_signal_power','observed_power')


def identities(counts,snrs):
    for part in COUNTS:
        for label in range(6):
            for snr in snrs:
                for frame in range(counts[part]):
                    for channel in (('A','B') if part=='test' else ('A',)):
                        yield FrameID(part,label,snr,frame),channel


def seal(folder,status,dependencies,**extra):
    folder=Path(folder)
    if (folder/'manifest.json').exists():raise ValueError('immutable manifest already exists')
    for name,digest in dependencies.items():
        if sha(name)!=digest:raise ValueError('source changed during execution')
    dump(folder/'manifest.json',dict(status=status,files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},dependencies=dependencies,**extra))


def verify(folder):
    folder=Path(folder).resolve();m=json.loads((folder/'manifest.json').read_text())
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p!=folder/'manifest.json'}
    if actual!=set(m['files']):raise ValueError('artifact closure differs')
    for name,digest in m['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or sha(p)!=digest:raise ValueError('artifact hash mismatch')
    for name,digest in m['dependencies'].items():
        p=Path(name).resolve()
        # Only local code/data closure or the one reviewed external baseline source.
        if not p.is_relative_to(ROOT) and p!=ROOT.parents[1]/'experiments/train_baselines.py':raise ValueError('dependency outside allowlist')
        if sha(p)!=digest:raise ValueError('dependency hash mismatch')
    return m


def assemble(output,*,counts=None,snr_indices=None,scientific=False,execution_config=None):
    counts=COUNTS.copy() if counts is None else dict(counts)
    snrs=list(range(20)) if snr_indices is None else list(snr_indices)
    if set(counts)!=set(COUNTS) or any(type(v)is not int or not 1<=v<=COUNTS[k] for k,v in counts.items()):raise ValueError('invalid split counts')
    if not snrs or sorted(set(snrs))!=snrs or any(type(s)is not int or not 0<=s<20 for s in snrs):raise ValueError('invalid SNR indices')
    if scientific:
        if execution_config is None:raise ValueError('formal assembly requires Phase11 gate')
        from .execution import check_gates
        check_gates(ROOT,execution_config)
        if counts!=COUNTS or snrs!=list(range(20)):raise ValueError('formal fixed matrix differs')
    elif 6*len(snrs)*(sum(counts.values())+counts['test'])>1000:
        raise ValueError('non-scientific smoke limited to 1000 observations')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dependencies={str(p):sha(p) for p in [Path(__file__),ROOT/'v2/phase12/core.py',ROOT/'configs/v2/phase12.yaml']}
    n=6*len(snrs)*(sum(counts.values())+counts['test'])
    maps={name:np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=dtype,shape=(n,*shape)) for name,(_,dtype,shape) in ARRAYS.items()}
    ids=np.empty(n,dtype='U80');waves=np.empty(n,dtype='U60');labels=np.empty(n,np.int64);db=np.empty(n,np.float32)
    parts=np.empty(n,dtype='U10');channels=np.empty(n,dtype='U1');frames=np.empty(n,np.int32);snrindex=np.empty(n,np.int16)
    scalar=np.empty((n,len(SCALARS)),np.float64);coeff=np.empty(n,np.complex128);keys=np.empty((n,len(PURPOSES),8),np.uint32)
    for row,(key,channel) in enumerate(identities(counts,snrs)):
        r=generate_frame(key,channel)
        for name,(field,_,_) in ARRAYS.items():maps[name][row]=r[field]
        ids[row]=key.name+':channel:'+channel;waves[row]=key.name;labels[row]=key.class_index;db[row]=r['target_snr_db']
        parts[row]=key.partition;channels[row]=channel;frames[row]=key.frame_index;snrindex[row]=key.snr_index
        scalar[row]=[r[name] for name in SCALARS];coeff[row]=r['channel_coefficient'];keys[row]=[r['stream_keys'][purpose] for purpose in PURPOSES]
    for array in maps.values():array.flush()
    maps.clear()
    splits={name+'_indices':np.flatnonzero(parts==name) for name in ('train','validation')}
    splits.update({f'test_{channel}_indices':np.flatnonzero((parts=='test')&(channels==channel)) for channel in ('A','B')})
    np.savez_compressed(output/'identity.npz',sample_ids=ids,waveform_ids=waves,labels=labels,snr_db=db,partitions=parts,channels=channels,frame_indices=frames,snr_indices=snrindex,**splits)
    np.savez_compressed(output/'latents.npz',scalar_values=scalar,scalar_names=np.array(SCALARS),channel_coefficients=coeff,stream_keys=keys,purpose_names=np.array(list(PURPOSES)),fir_coefficients=FIR)
    status='DATA_PREPARED_NOT_TRAINED' if scientific else 'NON_SCIENTIFIC_SMOKE'
    dump(output/'protocol.json',dict(status=status,counts=counts,snr_indices=snrs,observations=n,latent_frames=6*len(snrs)*sum(counts.values()),source_config_sha256=sha(ROOT/'configs/v2/phase12.yaml')))
    seal(output,status,dependencies)
    return output


def load_dataset(folder):
    folder=Path(folder)
    with np.load(folder/'identity.npz',allow_pickle=False) as b:identity={k:b[k] for k in b.files}
    return dict(folder=folder,identity=identity,**{name:np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ARRAYS})


def validate_dataset(folder,*,replay=False):
    m=verify(folder);data=load_dataset(folder);i=data['identity'];p=json.loads((Path(folder)/'protocol.json').read_text())
    if m['status']!=p['status'] or m['status'] not in ('DATA_PREPARED_NOT_TRAINED','NON_SCIENTIFIC_SMOKE'):raise ValueError('unrecognized dataset status')
    n=p['observations']
    if len(np.unique(i['sample_ids']))!=n or len(np.unique(i['waveform_ids']))!=p['latent_frames']:raise ValueError('identity uniqueness differs')
    if m['status']=='DATA_PREPARED_NOT_TRAINED' and (p['counts']!=COUNTS or p['snr_indices']!=list(range(20)) or n!=144000 or p['latent_frames']!=120000):raise ValueError('formal dataset matrix differs')
    for name,(_,dtype,shape) in ARRAYS.items():
        if data[name].shape!=(n,*shape) or data[name].dtype!=dtype:raise ValueError('array shape/dtype differs')
    for row,(key,ch) in enumerate(identities(p['counts'],p['snr_indices'])):
        if (i['sample_ids'][row]!=key.name+':channel:'+ch or i['waveform_ids'][row]!=key.name or i['labels'][row]!=key.class_index or i['snr_db'][row]!=-20+2*key.snr_index or i['partitions'][row]!=key.partition or i['channels'][row]!=ch):raise ValueError('row identity differs')
    expected_parts={name+'_indices':np.flatnonzero(i['partitions']==name) for name in ('train','validation')}
    expected_parts.update({f'test_{ch}_indices':np.flatnonzero((i['partitions']=='test')&(i['channels']==ch)) for ch in ('A','B')})
    for k,v in expected_parts.items():np.testing.assert_array_equal(i[k],v)
    np.testing.assert_array_equal(i['waveform_ids'][i['test_A_indices']],i['waveform_ids'][i['test_B_indices']])
    if replay:
        with np.load(Path(folder)/'latents.npz',allow_pickle=False) as b:
            scalar_values=b['scalar_values'];stream_keys=b['stream_keys'];coefficients=b['channel_coefficients']
            np.testing.assert_array_equal(b['fir_coefficients'],FIR)
            for row,(key,ch) in enumerate(identities(p['counts'],p['snr_indices'])):
                r=generate_frame(key,ch)
                for name,(field,_,_) in ARRAYS.items():np.testing.assert_array_equal(data[name][row],r[field])
                np.testing.assert_array_equal(scalar_values[row],[r[k] for k in SCALARS])
                np.testing.assert_array_equal(stream_keys[row],[r['stream_keys'][k] for k in PURPOSES])
                assert coefficients[row]==r['channel_coefficient']
    return dict(status='VALIDATED_DATA_PREPARATION',observations=n,latent_frames=p['latent_frames'],replayed=replay,scientific_results=False)
