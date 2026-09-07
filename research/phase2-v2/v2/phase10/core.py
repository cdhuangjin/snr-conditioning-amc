"""Prospective floor semantics, continuous diagnostics and seed-level comparisons."""
import numpy as np
from scipy.stats import t

from v2.phase8.core import conditions, head_logits, one_metrics

FLOORS = (None, -10, -8, -6)
STRATA = {'0_to_2': (0.,2.), '2_to_4': (2.,4.), '4_to_8': (4.,8.), '8_plus': (8.,np.inf)}
PAIRED_METRICS = ('accuracy','balanced_accuracy','macro_f1','concentration','prediction_entropy','error_to_dominant_share','raw_error_bias_db','raw_error_mae_db','raw_error_rmse_db')


def floor_conditions(source, truth, floor):
    source, truth = np.asarray(source), np.asarray(truth)
    if source.ndim != 1 or source.shape != truth.shape or not np.isfinite(source).all() or not np.isfinite(truth).all() or floor not in FLOORS:
        raise ValueError('invalid aligned raw conditions, truth or tested floor')
    clipped, _ = conditions(source)
    applied = clipped.copy() if floor is None else np.maximum(clipped, np.float32(floor))
    db, bins = conditions(applied)
    return {'source_condition_db':source.copy(),'range_clipped_condition_db':clipped,'applied_condition_db':db,'condition_bin':bins,'raw_error_db':source.astype(np.float64)-truth.astype(np.float64),'range_clipped_mask':(source < -20)|(source > 18),'floor_active_mask':np.zeros(len(source),dtype=bool) if floor is None else clipped < floor}


def _group(y, logits, raw_error, classes):
    if not len(y):
        return {**{name:None for name in PAIRED_METRICS},'count':0,'dominant_class':None,'confusion':None,'recall':None,'wrong_count':0,'errors_to_dominant_count':0,'raw_error_median_db':None}
    output = one_metrics(y, logits, classes)
    predicted = logits.argmax(1); wrong = predicted != y
    selected = int(np.sum(wrong & (predicted == output['dominant_class'])))
    output.update(wrong_count=int(wrong.sum()),errors_to_dominant_count=selected,error_to_dominant_share=float(selected/wrong.sum()) if wrong.any() else None,raw_error_bias_db=float(raw_error.mean()),raw_error_mae_db=float(np.abs(raw_error).mean()),raw_error_rmse_db=float(np.sqrt(np.mean(raw_error**2))),raw_error_median_db=float(np.median(raw_error)))
    return output


def metrics_with_strata(y, logits, snr, raw_error, classes=11):
    y, logits, snr, raw_error = np.asarray(y,dtype=np.int64), np.asarray(logits), np.asarray(snr), np.asarray(raw_error,dtype=float)
    if y.ndim != 1 or logits.shape != (len(y),classes) or snr.shape != y.shape or raw_error.shape != y.shape or np.any((y<0)|(y>=classes)) or not np.isfinite(logits).all() or not np.isfinite(snr).all() or not np.isfinite(raw_error).all():
        raise ValueError('finite row-aligned class/SNR/error/logit arrays required')
    def group(mask):
        return _group(y[mask],logits[mask],raw_error[mask],classes)
    absolute = np.abs(raw_error)
    return {'overall':group(np.ones(len(y),dtype=bool)), 'bands':{name:group(mask) for name,mask in {'low':snr<=-8,'mid':(snr>=-6)&(snr<=-2),'high':snr>=0}.items()},'snr':{str(db):group(snr==db) for db in range(-20,20,2)},'error_strata':{name:group((absolute>=low)&(absolute<high)) for name,(low,high) in STRATA.items()}}


def evaluate_floors(model, features, source, truth, floors, phase8_baseline, batch_size=64):
    if tuple(floors) != FLOORS:
        raise ValueError('all four prospectively tested floors are required')
    baseline = head_logits(model,features,source,batch_size)
    if baseline.shape != np.asarray(phase8_baseline).shape or not np.allclose(baseline,phase8_baseline,rtol=1e-6,atol=1e-7) or not np.array_equal(baseline.argmax(1),np.asarray(phase8_baseline).argmax(1)):
        raise ValueError('no-floor replay differs from Phase8 E1/E2')
    output = []
    for floor in floors:
        evidence = floor_conditions(source,truth,floor)
        score = baseline if floor is None else head_logits(model,features,evidence['applied_condition_db'],batch_size)
        unchanged = ~evidence['floor_active_mask']
        if not np.array_equal(score[unchanged],baseline[unchanged]):
            raise ValueError('unchanged above-floor conditions changed cached-head logits')
        output.append({**evidence,'floor_db':floor,'logits':score,'no_floor_phase8_max_abs':float(np.abs(baseline-np.asarray(phase8_baseline)).max()),'unchanged_rows':int(unchanged.sum()),'unchanged_logits_exact':True})
    return output


def _metric_groups(metrics):
    groups = {'overall':metrics['overall']}
    for section in ('bands','snr','error_strata'):
        groups.update({section+'/'+name:values for name,values in metrics[section].items()})
    return groups


def paired_rows(rows, models=('M3','M6'), conditions=('oracle','estimate')):
    index = {(r['model'],r['seed'],r['condition'],r['floor_db']):r for r in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate matched floor evaluation')
    results = []
    for model in models:
        for condition in conditions:
            for floor in FLOORS[1:]:
                pairs = [(_metric_groups(index[(model,seed,condition,floor)]['metrics']),_metric_groups(index[(model,seed,condition,None)]['metrics'])) for seed in range(2022,2027)]
                for group in pairs[0][0]:
                    for metric in PAIRED_METRICS:
                        if metric not in pairs[0][0][group]:
                            continue
                        values = [None if treatment[group][metric] is None or baseline[group][metric] is None else float(treatment[group][metric]-baseline[group][metric]) for treatment,baseline in pairs]
                        available = [v for v in values if v is not None]
                        if len(available)==5:
                            mean = float(np.mean(available)); sd = float(np.std(available,ddof=1)); half = float(t.ppf(.975,4)*sd/np.sqrt(5)); ci=[mean-half,mean+half]
                        else:
                            mean=sd=None;ci=None
                        results.append({'model':model,'condition':condition,'floor_db':floor,'baseline_floor_db':None,'group':group,'metric':metric,'seeds':list(range(2022,2027)),'differences':values,'n_pairs':len(available),'mean':mean,'std':sd,'ci95':ci,'missing_reason':None if len(available)==5 else 'requires_all_five_defined_seed_pairs','sampling_scope':'five training seeds on one fixed test partition'})
    return results
