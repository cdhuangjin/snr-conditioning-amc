import json
from pathlib import Path

import numpy as np
import pytest

from v2.phase5.core import generate, bayes_scores, decide, metrics, fit_erm, erm_scores
from v2.phase5.runner import validate_attempt


def setting(**kw):
    return dict(noise=1., asymmetry=1., global_gain=1., signal_gain=1., separation=1.,
                temperature=1., priors=[1/3]*3, rotation=0, tie_rule='first') | kw


def test_zero_information_uniform_bayes_exposes_tie_dependence():
    s = setting(separation=0)
    d = generate(s, 2022, 100, 9000)
    scores = bayes_scores(d['test_x'], d)
    np.testing.assert_array_equal(scores[:, 0], scores[:, 2])
    assert np.all(decide(scores, 'first', 2022) == 0)
    assert np.all(decide(scores, 'last', 2022) == 2)
    random = decide(scores, 'random', 2022)
    assert np.bincount(random, minlength=3).max()/len(random) < .36
    assert metrics(d['test_y'], random, np.full((9000, 3), 1/3))['balanced_accuracy'] == pytest.approx(1/3, abs=.02)


def test_global_gain_invariance_and_signal_gain_noninvariance():
    a = generate(setting(), 2022, 100, 100)
    b = generate(setting(global_gain=4), 2022, 100, 100)
    np.testing.assert_allclose(b['test_x'], 4*a['test_x'])
    sa, sb = bayes_scores(a['test_x'], a), bayes_scores(b['test_x'], b)
    np.testing.assert_allclose(sa-sa[:, :1], sb-sb[:, :1], atol=1e-12)
    c = generate(setting(signal_gain=4), 2022, 100, 100)
    assert not np.allclose(c['test_x'], b['test_x'])


def test_label_identity_is_pure_permutation():
    a = generate(setting(priors=[.6,.2,.2], asymmetry=4), 2022, 100, 100)
    b = generate(setting(priors=[.6,.2,.2], asymmetry=4, rotation=1), 2022, 100, 100)
    np.testing.assert_array_equal(a['test_x'], b['test_x'])
    np.testing.assert_array_equal((a['test_y']+1)%3, b['test_y'])
    np.testing.assert_allclose(bayes_scores(b['test_x'], b), np.roll(bayes_scores(a['test_x'], a), 1, axis=1))


def test_erm_fits_training_data_only_and_replays():
    d = generate(setting(), 2022, 500, 100)
    opts = dict(l2=.0001,maxiter=1000,gtol=1e-8,ftol=1e-12)
    model = fit_erm(d['train_x'], d['train_y'], opts)
    assert model['converged']
    np.testing.assert_allclose(model['mean'], d['train_x'].mean(0))
    score = erm_scores(d['test_x'], model)
    d['test_x'][:] = 1e10
    again = fit_erm(d['train_x'], d['train_y'], opts)
    np.testing.assert_array_equal(model['weights'], again['weights'])
    assert score.shape == (100,3)


def test_metric_limits_and_temperature_cannot_change_argmax():
    y = np.tile(np.arange(3), 4)
    p = np.full((12,3), 1/3)
    m = metrics(y, np.zeros(12,dtype=int), p)
    assert m['concentration'] == 1 and m['prediction_entropy'] == 0
    assert m['balanced_accuracy'] == pytest.approx(1/3)
    scores = np.array([[1.,3.,2.],[-1.,0.,1.]])
    for t in [.25,1,4]:
        np.testing.assert_array_equal(decide(scores/t,'first',2022), [1,2])


def test_manifest_tampering_rejected_before_numeric_loading(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    manifest = {'status':'COMPLETE','files':{'config.json':'0'*64}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='hash'):
        validate_attempt(tmp_path)
