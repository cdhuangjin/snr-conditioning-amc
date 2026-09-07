"""Dataset-specific conditions and bounded waveform access."""
from pathlib import Path
import numpy as np
import torch
from v2.phase6.audit import frame_features


def dynamic_conditions(raw,grid):
    grid=np.asarray(grid,dtype=np.float32);raw=np.asarray(raw)
    if grid.ndim!=1 or len(grid)<2 or np.any(np.diff(grid)<=0) or raw.ndim!=1 or not np.isfinite(raw).all():raise ValueError('invalid dynamic condition grid')
    db=np.clip(raw,grid[0],grid[-1]).astype(np.float32)
    return db,np.abs(db[:,None]-grid).argmin(1).astype(np.int64)


class LazyFrames(torch.utils.data.Dataset):
    def __init__(self,signals,labels,condition,indices,grid):
        self.signals=signals;self.labels=np.asarray(labels);self.condition=np.asarray(condition);self.indices=np.asarray(indices,dtype=np.int64);self.grid=np.asarray(grid)
        if signals.shape[1]!=2 or signals.ndim!=3 or signals.dtype!=np.float32 or len(signals)!=len(labels) or len(labels)!=len(condition):raise ValueError('unaligned NCT float32 data')
        if np.any(self.indices<0) or np.any(self.indices>=len(labels)):raise ValueError('row outside source')
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        row=self.indices[index];db,bins=dynamic_conditions(self.condition[row:row+1],self.grid)
        # Copy only this frame: mmap is read-only and Torch must not mutate it.
        return torch.from_numpy(np.array(self.signals[row],copy=True)),int(self.labels[row]),np.float32(db[0]),np.int64(bins[0])


def bounded_features(signals,chunk_size=128,feature_fn=frame_features):
    if chunk_size<1 or chunk_size>1024:raise ValueError('bounded feature chunk required')
    result=np.empty((len(signals),7),dtype=np.float64)
    for start in range(0,len(signals),chunk_size):result[start:start+chunk_size]=feature_fn(signals[start:start+chunk_size])
    if not np.isfinite(result).all():raise ValueError('nonfinite frame statistics')
    return result


def validate_identity(identity):
    n=len(identity['labels']);parts=[identity[name+'_indices'] for name in ['train','validation','test']]
    flat=np.concatenate(parts)
    if len(np.unique(flat))!=len(flat) or np.any(flat<0) or np.any(flat>=n):raise ValueError('partition overlap or row escape')
    expected=identity.get('retained_indices',np.arange(n))
    if not np.array_equal(np.sort(flat),np.sort(expected)):raise ValueError('split does not cover retained rows')
    if any(not len(p) for p in parts):raise ValueError('empty partition')
    return parts


def load_dataset(root,name):
    import importlib.util
    import pickle
    from v2.phase11.subset import validate_subset
    from v2.phase8.evidence import load
    root=Path(root).resolve()
    if name=='2018':
        folder=root/'results/v2/phase11_preflight/2018_subset';source=root.parents[1]/'data/GOLD_XYZ_OSC.0001_1024.hdf5'
        validate_subset(folder,source,root,replay_source=False)
        signals=np.load(folder/'signals.npy',mmap_mode='r',allow_pickle=False);classes=24
    elif name=='04C':
        spec=importlib.util.spec_from_file_location('phase11_fixed_04c',root/'scripts/v2/prepare_04c_split.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.validate()
        folder=module.OUTPUT;source=module.SOURCE
        # Trusted, SHA-verified local mirror pickle. At most 166 MB of waveforms.
        with source.open('rb') as handle:raw=pickle.load(handle,encoding='latin1')
        keys=sorted(raw);signals=np.concatenate([raw[key] for key in keys]).astype(np.float32,copy=False);classes=11
        with np.load(folder/'identity.npz',allow_pickle=False) as b:
            class_names=b['class_names'].tolist();labels=np.concatenate([np.full(len(raw[key]),class_names.index(key[0].decode('ascii') if isinstance(key[0],bytes) else key[0])) for key in keys]);snr=np.concatenate([np.full(len(raw[key]),key[1]) for key in keys])
            np.testing.assert_array_equal(labels,b['labels']);np.testing.assert_array_equal(snr,b['snr_db'])
    else:raise ValueError('unknown cross-dataset source')
    with np.load(folder/'identity.npz',allow_pickle=False) as b:identity={k:b[k] for k in b.files}
    validate_identity(identity);grid=load(folder/'config.json')['grid']
    if signals.shape!=(len(identity['labels']),2,1024 if name=='2018' else 128):raise ValueError('source waveform shape differs')
    return dict(name=name,signals=signals,identity=identity,grid=grid,classes=classes,ids=np.array([f'{name}:source:{r}' for r in identity['source_rows']]),source=source,preparation=folder)
