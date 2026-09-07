"""Known Gaussian mixtures and train-only multinomial cross-entropy ERM."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def generate(setting, seed, n_train, n_test):
    priors = np.asarray(setting['priors'], dtype=float)
    if priors.shape != (3,) or np.any(priors <= 0) or not np.isclose(priors.sum(), 1):
        raise ValueError('three positive normalized priors required')
    for key in ('noise','asymmetry','global_gain','signal_gain','temperature'):
        if not np.isfinite(setting[key]) or setting[key] <= 0:
            raise ValueError(f'{key} must be positive finite')
    if setting['separation'] < 0 or not np.isfinite(setting['separation']):
        raise ValueError('separation must be nonnegative finite')
    if setting['rotation'] not in (0,1,2) or setting['tie_rule'] not in ('first','last','random'):
        raise ValueError('invalid label rotation or tie rule')
    centers = np.array([[1.,0.],[-.5,np.sqrt(3)/2],[-.5,-np.sqrt(3)/2]])
    means = centers * setting['separation'] * setting['signal_gain'] * setting['global_gain']
    std = np.array([setting['asymmetry'],1.,1.])*setting['noise']*setting['global_gain']
    rotation = setting['rotation']
    result = dict(means=np.roll(means,rotation,axis=0),std=np.roll(std,rotation),priors=np.roll(priors,rotation))
    for partition, n, stream in [('train',n_train,0),('test',n_test,1)]:
        rng = np.random.default_rng(np.random.SeedSequence([seed,stream]))
        latent = rng.choice(3,size=n,p=priors)
        result[partition+'_x'] = means[latent] + std[latent,None]*rng.normal(size=(n,2))
        result[partition+'_y'] = (latent+rotation)%3
        result[partition+'_ids'] = np.array([f'{seed}:{partition}:{i}' for i in range(n)])
    return result


def bayes_scores(x, distribution):
    delta = x[:,None,:]-distribution['means'][None,:,:]
    variance = distribution['std']**2
    return np.log(distribution['priors'])[None,:] - np.log(2*np.pi*variance)[None,:] - np.sum(delta**2,axis=2)/(2*variance[None,:])


def decide(scores, rule, seed):
    tied = scores == scores.max(axis=1,keepdims=True)
    if rule == 'first': return np.argmax(tied,axis=1)
    if rule == 'last': return 2-np.argmax(tied[:,::-1],axis=1)
    if rule != 'random': raise ValueError('unknown tie rule')
    rng = np.random.default_rng(np.random.SeedSequence([seed,2]))
    # Independent continuous random ranks yield a uniform selection among exact maxima.
    return np.argmax(np.where(tied,rng.random(scores.shape),-1),axis=1)


def fit_erm(x, y, options):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale == 0,1.,scale)
    z = np.column_stack([(x-mean)/scale,np.ones(len(x))])
    target = np.eye(3)[y]
    def objective(flat):
        w = flat.reshape(3,3)
        logits = z@w
        loss = np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])
        loss += options['l2']/2*np.sum(w[:2]**2)
        gradient = z.T@(softmax(logits,axis=1)-target)/len(y)
        gradient[:2] += options['l2']*w[:2]
        return float(loss), gradient.ravel()
    opt = minimize(objective,np.zeros(9),method='L-BFGS-B',jac=True,
                   options={key:options[key] for key in ('maxiter','gtol','ftol')})
    return dict(weights=opt.x.reshape(3,3),mean=mean,scale=scale,converged=bool(opt.success),
                iterations=int(opt.nit),objective=float(opt.fun),gradient_inf=float(np.max(np.abs(opt.jac))),message=str(opt.message))


def erm_scores(x, model):
    return np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))])@model['weights']


def metrics(y, predictions, probabilities):
    counts = np.bincount(predictions,minlength=3)
    share = counts/len(y)
    nonzero = share[share>0]
    support = np.bincount(y,minlength=3)
    if np.any(support == 0): raise ValueError('all true classes required for BA')
    recall = [float(np.mean(predictions[y==c]==c)) for c in range(3)]
    posterior_entropy = -np.sum(probabilities*np.log(np.clip(probabilities,1e-300,1)),axis=1)/np.log(3)
    return dict(accuracy=float(np.mean(y==predictions)),balanced_accuracy=float(np.mean(recall)),recall=recall,
                concentration=float(share.max()),prediction_entropy=float(-np.sum(nonzero*np.log(nonzero))/np.log(3)),
                dominant_class=int(np.argmax(counts)),predicted_counts=counts.tolist(),true_counts=support.tolist(),
                mean_posterior_entropy=float(posterior_entropy.mean()),nll=float(-np.log(np.clip(probabilities[np.arange(len(y)),y],1e-300,1)).mean()))
