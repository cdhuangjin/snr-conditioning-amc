"""Array-derived classification, estimator, pilot cost and paired seed summaries."""
import numpy as np
from scipy.special import softmax
from scipy.stats import t
from v2.phase8.core import one_metrics
from .core import overhead


def masks(labels,snr):
    return {'overall':np.ones(len(labels),bool),**{'band:'+k:v for k,v in {'low':snr<=-8,'mid':(snr>=-6)&(snr<=-2),'high':snr>=0}.items()},**{f'snr:{int(s)}':snr==s for s in np.unique(snr)},**{f'class:{int(c)}':labels==c for c in range(6)}}


def summarize(labels,logits,snr,choice):
    result={};pred=logits.argmax(1);prob=softmax(logits.astype(float),axis=1)
    for name,mask in masks(labels,snr).items():
        if not mask.any():result[name]=None;continue
        y=labels[mask];p=pred[mask];pr=prob[mask];correct=p==y;wrong=~correct
        m=one_metrics(y,logits[mask],6)
        m['error_to_dominant_share']=float(np.mean(p[wrong]==m['dominant_class'])) if wrong.any() else None
        m['nll']=float(-np.log(np.maximum(pr[np.arange(len(y)),y],1e-300)).mean());confidence=pr.max(1);ece=0.
        for lower in range(15):
            selected=(confidence>=lower/15)&((confidence<(lower+1)/15) if lower<14 else confidence<=1)
            if selected.any():ece+=selected.mean()*abs(confidence[selected].mean()-correct[selected].mean())
        m['ece15']=float(ece)
        m.update(overhead(0 if choice['K'] is None else choice['K'],m['accuracy']))
        m['K']=choice['K'];m['estimator']=choice['estimator']
        for key,values in [('guarded',choice['raw_db']),('clipped',choice['clipped_db'])]:
            err=np.asarray(values)[mask]-snr[mask]
            m[key+'_bias_db']=float(err.mean());m[key+'_mae_db']=float(abs(err).mean());m[key+'_rmse_db']=float(np.sqrt(np.mean(err**2)))
        if 'raw_db_valid' in choice:
            valid=np.asarray(choice['raw_db_valid'])[mask];raw=np.asarray(choice['raw_pilot_db'])[mask];error=raw[valid]-snr[mask][valid]
            m['raw_valid_count']=int(valid.sum());m['raw_valid_only_mae_db']=float(abs(error).mean()) if valid.any() else None
            m['raw_valid_only_rmse_db']=float(np.sqrt(np.mean(error**2))) if valid.any() else None
            m['guard_counts']={k:int(np.asarray(choice[k])[mask].sum()) for k in ['zero_noise_guard','nonpositive_signal_guard','noise_floor_guard','signal_floor_guard']}
        result[name]=m
    return result


def paired(rows,seeds):
    index={(r['model'],r['seed'],r['channel'],r['condition']):r for r in rows};out=[]
    conditions=sorted({r['condition'] for r in rows});groups=sorted(rows[0]['metrics'].keys())
    metrics=('accuracy','balanced_accuracy','macro_f1','concentration','prediction_entropy','nll','ece15','utility_proxy')
    def append(kind,left,right,group,metric,meta):
        values=[]
        for seed in seeds:
            a=index[left(seed)]['metrics'][group];b=index[right(seed)]['metrics'][group]
            if a is None or b is None:continue
            values.append(float(a[metric]-b[metric]))
        if len(values)!=len(seeds):return
        mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if len(values)>1 else None
        half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))) if sd is not None else None
        out.append(dict(kind=kind,**meta,group=group,metric=metric,seeds=seeds,differences=values,mean=mean,std=sd,ci95=None if half is None else [mean-half,mean+half],scope='network seeds on one fixed paired payload dataset; not independent waveform repetitions'))
    for condition in conditions:
        for group in groups:
            for metric in metrics:
                for channel in ('A','B'):
                    for baseline in ('M0','CLDNN'):
                        append('M6_minus_baseline',lambda seed:('M6',seed,channel,condition),lambda seed:(baseline,seed,channel,condition),group,metric,dict(condition=condition,channel=channel,baseline=baseline))
                for model in ('M0','M6','CLDNN'):
                    append('paired_B_minus_A',lambda seed:(model,seed,'B',condition),lambda seed:(model,seed,'A',condition),group,metric,dict(condition=condition,model=model))
    return out


def figures(folder,rows,scientific):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from v2.phase8.evidence import dump,sha
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,channel in zip(axes,('A','B')):
        for model in ('M0','M6','CLDNN'):
            values=[]
            for k in (0,8,16,32,64):
                name='K0_frame_estimator' if k==0 else f'K{k}_pilot'
                chosen=[r['metrics']['overall']['utility_proxy'] for r in rows if r['channel']==channel and r['model']==model and r['condition']==name]
                values.append(np.mean(chosen))
            ax.plot([0,8,16,32,64],values,marker='o',label=model)
        ax.set(xlabel='Additional pilot samples (K=0: frame fallback)',ylabel='Payload fraction × accuracy',title=f'Channel {channel}');ax.legend()
    fig.suptitle('Pilot cost utility proxy' if scientific else 'NON-SCIENTIFIC SMALL SMOKE')
    fig.tight_layout();fig.savefig(folder/'pilot_utility.png',dpi=180);fig.savefig(folder/'pilot_utility.pdf');plt.close(fig)
    dump(folder/'figure_sources.json',{'rows.json':sha(folder/'rows.json'),'paired.json':sha(folder/'paired.json')})
