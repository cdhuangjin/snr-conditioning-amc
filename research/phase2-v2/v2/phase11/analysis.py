"""Unequal-cell, calibration and paired cross-dataset reporting."""
import numpy as np
from scipy.special import softmax
from scipy.stats import t
from v2.phase8.core import one_metrics


def metrics(y,logits,snr,classes):
    y=np.asarray(y);snr=np.asarray(snr);logits=np.asarray(logits)
    def group(mask):
        target=y[mask];score=logits[mask];db=snr[mask]
        if not len(target):return None
        result=one_metrics(target,score,classes);prediction=score.argmax(1);wrong=prediction!=target
        cells=[np.mean(prediction[(target==c)&(db==s)]==c) for c,s in sorted(set(zip(target.tolist(),db.tolist())))]
        probability=softmax(score.astype(float),axis=1);confidence=probability.max(1);correct=prediction==target
        ece=0.
        for i in range(15):
            selected=(confidence>=i/15)&((confidence<(i+1)/15) if i<14 else (confidence<=1))
            if selected.any():ece+=selected.mean()*abs(confidence[selected].mean()-correct[selected].mean())
        result.update(equal_cell_accuracy=float(np.mean(cells)),sample_weighted_accuracy=float(correct.mean()),nll=float(-np.log(np.maximum(probability[np.arange(len(target)),target],1e-300)).mean()),ece15=float(ece),error_to_dominant_share=float(np.mean(prediction[wrong]==result['dominant_class'])) if wrong.any() else None)
        return result
    return dict(overall=group(np.ones(len(y),bool)),snr={str(int(s)):group(snr==s) for s in np.unique(snr)},bands={name:group(mask) for name,mask in {'low':snr<=-8,'mid':(snr>=-6)&(snr<=-2),'high':snr>=0}.items()})


def paired(rows):
    index={(r['dataset'],r['model'],r['seed'],r['condition']):r for r in rows};output=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for condition in ['oracle','estimated']:
            for group in ['overall','low','mid','high']:
                for metric in ['accuracy','balanced_accuracy','equal_cell_accuracy','concentration','prediction_entropy','nll','ece15']:
                    def value(row):return (row['metrics']['overall'] if group=='overall' else row['metrics']['bands'][group])[metric]
                    values=np.array([value(index[(dataset,'M6',seed,condition)])-value(index[(dataset,'M0',seed,'agnostic')]) for seed in range(2022,2027)])
                    mean=float(values.mean());sd=float(values.std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5))
                    output.append(dict(dataset=dataset,condition=condition,baseline='M0',group=group,metric=metric,seeds=list(range(2022,2027)),n_pairs=5,differences=values.tolist(),mean=mean,std=sd,ci95=[mean-half,mean+half],scope='five network seeds on one fixed dataset partition'))
    return output


def figures(attempt,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from v2.phase8.evidence import dump,sha
    datasets=sorted({r['dataset'] for r in rows})
    fig,axes=plt.subplots(1,len(datasets),figsize=(5*len(datasets),4))
    for ax,dataset in zip(np.atleast_1d(axes),datasets):
        for model,condition in [('M0','agnostic'),('M6','oracle'),('M6','estimated')]:
            selected=[r for r in rows if r['dataset']==dataset and r['model']==model and r['condition']==condition];grid=sorted(int(s) for s in selected[0]['metrics']['snr'])
            values=np.array([[r['metrics']['snr'][str(s)]['balanced_accuracy'] for r in selected] for s in grid])
            ax.errorbar(grid,values.mean(1),yerr=values.std(1,ddof=1),label=f'{model}/{condition}',capsize=2)
        ax.set(title=dataset,xlabel='True SNR (dB)',ylabel='Balanced accuracy');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(attempt/'cross_dataset.png',dpi=300);fig.savefig(attempt/'cross_dataset.pdf');plt.close(fig)
    dump(attempt/'figure_sources.json',{'rows.json':sha(attempt/'rows.json'),'paired.json':sha(attempt/'paired.json')})
