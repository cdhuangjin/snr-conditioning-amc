"""Saved-prediction analyses; no fitting or test-driven hyperparameter selection."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from v2.phase8.core import conditions,grouped_metrics,head_logits
from v2.phase8.evidence import dump,load,sha


def error_strata(true_snr,estimated,labels,dataset='RML2016.10a',channel='in_dataset_unspecified'):
    error=np.asarray(estimated)-np.asarray(true_snr)
    def stats(mask):
        e=error[mask]
        return dict(n=len(e),mean=float(e.mean()),std=float(e.std(ddof=1)) if len(e)>1 else 0.,mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),quantiles=np.quantile(e,[0,.05,.5,.95,1]).tolist())
    return dict(dataset=dataset,channel=channel,overall=stats(np.ones(len(error),dtype=bool)),snr={str(float(s)):stats(true_snr==s) for s in np.unique(true_snr)},modulation={str(int(c)):stats(labels==c) for c in np.unique(labels)})


def matched_statistics(rows):
    index={(r['arm'],r['seed'],r['deployment']):r for r in rows if r['role']=='matched_primary'}
    result=[]
    for treatment in ('generic','empirical_snr'):
        for baseline in (['clean'] if treatment=='generic' else ['clean','generic']):
            for metric in ('accuracy','balanced_accuracy','concentration','prediction_entropy'):
                delta=np.array([index[(treatment,s,'estimated')]['metrics']['overall'][metric]-index[(baseline,s,'estimated')]['metrics']['overall'][metric] for s in range(2022,2027)])
                half=float(t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5))
                result.append(dict(treatment=treatment,baseline=baseline,deployment='estimated',metric=metric,seeds=list(range(2022,2027)),n_pairs=5,differences=delta.tolist(),mean=float(delta.mean()),std=float(delta.std(ddof=1)),ci95=[float(delta.mean()-half),float(delta.mean()+half)],scope='network seeds conditional on one fixed split/estimator/OOF pool'))
    return result


def cliff_specs(config):
    rows=[]
    for model in config['cliff']['models']:
        for seed in config['seeds']:
            for origin in config['cliff']['origins']:
                for delta in config['cliff']['deltas_db']:
                    rows.append(dict(model=model,seed=seed,origin=origin,kind='additive',value=float(delta),id=f'{model}_{seed}_{origin}_delta{delta}'))
            if model=='M3':
                for boundary in range(-19,18,2):
                    for offset in (-config['cliff']['boundary_epsilon_db'],0.,config['cliff']['boundary_epsilon_db']):
                        rows.append(dict(model=model,seed=seed,origin='oracle',kind='boundary',value=float(boundary+offset),boundary_db=boundary,offset_db=float(offset),id=f'{model}_{seed}_boundary{boundary}_{offset}'))
    return rows


def parameter_distance(model,base,changed):
    base_db,base_bin=conditions(base);db,bins=conditions(changed)
    with torch.no_grad():
        if model.conditioning=='M3':
            delta=model.snr_embedding(torch.as_tensor(bins))-model.snr_embedding(torch.as_tensor(base_bin));name='embedding_l2'
        else:
            a=model.normalize_snr(torch.as_tensor(base_db))[:,None];b=model.normalize_snr(torch.as_tensor(db))[:,None]
            delta=2*torch.sigmoid(model.gate(b))-2*torch.sigmoid(model.gate(a));name='gate_l2'
    return name,torch.linalg.vector_norm(delta,dim=1).numpy()


def run_cliffs(attempt,phase8_attempt,phase3,p3config,runs,dataset,split,estimated,config,*,verify_only=False):
    from v2.phase8.runner import get_model
    output=attempt/'cliffs'
    if not verify_only:output.mkdir(exist_ok=True)
    rows=[];test=split.test_idx;y=dataset['labels'][test];truth=dataset['snrs'][test];ids=split.sample_ids[test]
    specs=cliff_specs(config)
    for model_id in config['cliff']['models']:
        for seed in config['seeds']:
            model=get_model(phase3,p3config,runs[(model_id,seed)],model_id,'cpu')
            with np.load(phase8_attempt/'caches'/f'{model_id}_{seed}.npz',allow_pickle=False) as b:
                cache=b['raw_pooled_features'];np.testing.assert_array_equal(b['sample_ids'],ids)
            origins={'oracle':truth,'estimated':conditions(estimated[test])[0]}
            baseline={origin:head_logits(model,cache,value,64) for origin,value in origins.items()}
            for spec in [s for s in specs if s['model']==model_id and s['seed']==seed]:
                base=origins[spec['origin']];raw=base+spec['value'] if spec['kind']=='additive' else np.full(len(base),spec['value'])
                db,bins=conditions(raw);score=head_logits(model,cache,db,64);name,distance=parameter_distance(model,base,db)
                path=output/(spec['id']+'.npz')
                arrays=dict(sample_ids=ids,y_true=y,snr_db=truth,origin_condition_db=base,raw_condition_db=raw,deployed_condition_db=db,condition_bin=bins,logits=score,y_pred=score.argmax(1),parameter_distance=distance)
                if path.exists():
                    with np.load(path,allow_pickle=False) as old:
                        for key,value in arrays.items():np.testing.assert_array_equal(old[key],value)
                elif verify_only:raise ValueError('cliff source bundle missing')
                else:np.savez_compressed(path,**arrays)
                metrics=grouped_metrics(y,score,truth);base_metrics=grouped_metrics(y,baseline[spec['origin']],truth)
                rows.append(spec|dict(bundle=path.relative_to(attempt).as_posix(),bundle_sha256=sha(path),metrics=metrics,balanced_accuracy_delta=metrics['overall']['balanced_accuracy']-base_metrics['overall']['balanced_accuracy'],prediction_disagreement=float(np.mean(score.argmax(1)!=baseline[spec['origin']].argmax(1))),saturation_fraction=float(np.mean((raw<-20)|(raw>18))),exact_midpoint_fraction=float(np.mean((db+19)%2==0)),bin_crossing_fraction=float(np.mean(bins!=conditions(base)[1])),parameter_distance_kind=name,parameter_distance_mean=float(distance.mean()),diagnostic_not_deployment=spec['kind']=='boundary'))
    if len(rows)!=545:raise ValueError('incomplete cliff matrix')
    if verify_only:
        if load(attempt/'cliff_rows.json')!=rows:raise ValueError('cliff head/metric/parameter-distance replay differs')
    else:dump(attempt/'cliff_rows.json',rows)
    return rows


def figures(attempt,rows,cliffs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,metric in zip(axes,['accuracy','balanced_accuracy','concentration']):
        for i,arm in enumerate(['clean','generic','empirical_snr']):
            values=[r['metrics']['overall'][metric] for r in rows if r['role']=='matched_primary' and r['deployment']=='estimated' and r['arm']==arm]
            ax.scatter(np.full(5,i),values,s=15);ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),fmt='ko',capsize=4)
        ax.set_xticks(range(3),['clean','generic','empirical SNR'],rotation=20);ax.set_ylabel(metric)
    fig.suptitle('Matched new training: estimated deployment, five seeds');fig.tight_layout();fig.savefig(attempt/'training_comparison.png',dpi=300);fig.savefig(attempt/'training_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for model,color in [('M3','#0072B2'),('M6','#D55E00')]:
        for origin,style in [('oracle','-'),('estimated','--')]:
            selected=[r for r in cliffs if r['model']==model and r['origin']==origin and r['kind']=='additive'];values=sorted({r['value'] for r in selected})
            axes[0].plot(values,[np.mean([r['metrics']['overall']['balanced_accuracy'] for r in selected if r['value']==v]) for v in values],style,color=color,label=f'{model}/{origin}')
            axes[1].plot(values,[np.mean([r['prediction_disagreement'] for r in selected if r['value']==v]) for v in values],style,color=color)
            axes[2].scatter([r['parameter_distance_mean'] for r in selected],[r['balanced_accuracy_delta'] for r in selected],color=color,s=8,alpha=.4)
    axes[0].legend(fontsize=7);axes[0].set(xlabel='Added SNR error (dB)',ylabel='Balanced accuracy');axes[1].set(xlabel='Added SNR error (dB)',ylabel='Prediction disagreement');axes[2].set(xlabel='Embedding (M3) / gate (M6) L2 change',ylabel='Balanced accuracy change')
    fig.tight_layout();fig.savefig(attempt/'condition_cliffs.png',dpi=300);fig.savefig(attempt/'condition_cliffs.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    boundaries=sorted({r['boundary_db'] for r in cliffs if r['kind']=='boundary'})
    for offset in [-.001,0.,.001]:
        selected=[r for r in cliffs if r['kind']=='boundary' and r['offset_db']==offset]
        for ax,metric in zip(axes,['balanced_accuracy','concentration']):
            values=np.array([[r['metrics']['overall'][metric] for r in selected if r['boundary_db']==b] for b in boundaries])
            ax.errorbar(boundaries,values.mean(1),yerr=values.std(1,ddof=1),label=f'b {offset:+g} dB',capsize=2)
            ax.set(xlabel='M3 bin boundary b (dB)',ylabel=metric.replace('_',' '))
    axes[0].legend();fig.suptitle('Constant-condition boundary interventions; mean ± SD over five seeds')
    fig.tight_layout();fig.savefig(attempt/'bin_transition_sensitivity.png',dpi=300);fig.savefig(attempt/'bin_transition_sensitivity.pdf');plt.close(fig)
    dump(attempt/'figure_sources.json',{'rows.json':sha(attempt/'rows.json'),'cliff_rows.json':sha(attempt/'cliff_rows.json'),'interpretation':'parameter-distance association is descriptive, not a causal regression'})
