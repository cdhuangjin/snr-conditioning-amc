import inspect
import numpy as np
import pytest

from v2.phase12.core import (FrameID, constellation, clean_payload, generate_frame,
    proper_noise, channel_observation, pilot_estimate, overhead, pilot_sequence)


def test_constellation_energy_and_uniform_draw():
    for label, count in enumerate([2, 4, 8, 16]):
        points = constellation(label)
        assert len(points) == count
        assert np.mean(abs(points)**2) == pytest.approx(1)
        rng = np.random.default_rng(7)
        counts = np.bincount(rng.integers(count, size=160000), minlength=count)
        assert np.max(abs(counts / 160000 - 1/count)) < .005


def test_determinism_identity_separation_and_unit_power():
    for label in range(6):
        key = FrameID('test', label, 10, 0)
        a, b = generate_frame(key, 'A'), generate_frame(key, 'A')
        assert np.array_equal(a['payload'], b['payload'])
        assert np.mean(abs(a['clean'])**2) == pytest.approx(1, abs=1e-12)
        assert a['nct'].shape == (2, 128) and a['nct'].dtype == np.float32
        assert not np.array_equal(a['clean'], clean_payload(FrameID('train', label, 10, 0)))
        paired = generate_frame(key, 'B')
        assert np.array_equal(a['clean'], paired['clean'])
        assert a['stream_keys']['payload_noise'] != paired['stream_keys']['payload_noise']
    with pytest.raises(ValueError):
        generate_frame(FrameID('train', 0, 0, 0), 'B')


def test_noise_distribution_and_gain_equivariance():
    n = proper_noise(np.random.default_rng(23), (200000,), 3.)
    assert np.mean(abs(n)**2) == pytest.approx(3., rel=.01)
    assert np.mean(n.real**2) == pytest.approx(1.5, rel=.01)
    assert abs(np.mean(n)) < .015
    clean = np.ones(128, complex)
    t = np.arange(128)
    a = channel_observation(clean, t, .3+.4j, 0., 1., n[:128])
    b = channel_observation(clean, t, .3+.4j, 0., 2., n[:128])
    assert np.array_equal(b, 2*a)
    assert np.allclose(a, (.3+.4j)*clean+n[:128])


def test_pilot_times_suffix_and_cfo():
    result = generate_frame(FrameID('test', 0, 0, 1), 'B')
    p = pilot_sequence()
    h, f, g = result['channel_coefficient'], result['cfo'], result['gain_amplitude']
    expected = g*h*p*np.exp(2j*np.pi*f*np.arange(-64, 0))
    assert np.allclose(result['pilot']-g*result['pilot_noise'], expected)
    for k in [8, 16, 32, 64]:
        fitted = pilot_estimate(result['pilot'][-k:], p[-k:])
        assert fitted['pilot_count'] == k
    assert np.allclose(channel_observation(p, np.arange(-64,0), h, 0., g, np.zeros(64)), g*h*p)


def test_ls_complex_degrees_of_freedom_and_consistency():
    rng = np.random.default_rng(9)
    p = pilot_sequence()
    h, sigma = .8+.6j, .25
    y = h*p + proper_noise(rng, (20000,64), sigma)
    estimate = pilot_estimate(y, p)
    assert np.mean(estimate['noise_variance']) == pytest.approx(sigma, rel=.004)
    assert np.mean(estimate['corrected_signal_power']) == pytest.approx(abs(h)**2, rel=.004)
    direct_h = (y*p.conj()).mean(axis=-1)
    direct_noise = np.sum(abs(y-direct_h[:,None]*p)**2,axis=-1)/63
    assert np.array_equal(estimate['noise_variance'], direct_noise)
    assert 'snr' not in inspect.signature(pilot_estimate).parameters


def test_guards_and_k_zero():
    p = pilot_sequence(8)
    z = pilot_estimate(np.zeros(8, complex), p)
    assert z['zero_noise_guard'] and z['nonpositive_signal_guard']
    assert not z['raw_db_valid'] and np.isnan(z['signed_linear_snr'])
    assert np.isfinite(z['clipped_db'])
    neg = pilot_estimate(p*np.array([1,-1]*4), p)
    assert neg['signed_linear_snr'] < 0 and not neg['raw_db_valid']
    with pytest.raises(ValueError, match='fallback'):
        pilot_estimate(np.empty(0), np.empty(0))
    with pytest.raises(ValueError):
        pilot_estimate(np.ones(8)*np.nan, p)
    for k in [0,8,16,32,64]:
        o = overhead(k, .5)
        assert o['payload_fraction'] == 128/(128+k)
        assert o['utility_proxy'] == .5*128/(128+k)


def test_identity_limits():
    for args in [('train',6,0,0),('test',0,20,0),('test',0,0,200)]:
        with pytest.raises(ValueError):
            FrameID(*args)


def test_order_future_noninterference_and_expected_snr():
    keys = [FrameID('test', c, s, 4) for c in range(6) for s in (0,19)]
    first = {k.name:generate_frame(k, 'B') for k in keys}
    generate_frame(FrameID('test',5,13,99), 'B')
    for k in reversed(keys):
        a, b = first[k.name], generate_frame(k,'B')
        assert np.array_equal(a['payload'],b['payload'])
        assert np.array_equal(a['pilot'],b['pilot'])
        assert 10*np.log10(a['received_expected_signal_power']/a['noise_variance_after_gain']) == pytest.approx(a['target_snr_db'])
        assert np.mean(abs(a['payload_noise'])**2) != a['noise_variance_before_gain']


def test_analog_filter_and_frozen_config():
    from pathlib import Path
    import yaml
    from v2.phase12.core import FIR, VERSION, SEED, COUNTS, PARTITIONS, CHANNELS, PURPOSES
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v2/phase12.yaml').read_text())
    assert cfg['generator_version'] == VERSION and cfg['construction_seed'] == SEED
    assert cfg['frames_per_cell'] == COUNTS and cfg['partition_codes'] == PARTITIONS
    assert cfg['channel_codes'] == CHANNELS and cfg['purpose_codes'] == PURPOSES
    assert len(FIR) == cfg['fir']['taps'] and FIR.sum() == pytest.approx(1)
    assert np.allclose(FIR,FIR[::-1])
    key = FrameID('train',4,0,0)
    white = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key.stream_key('clean','message')))).standard_normal(192)
    message = np.convolve(white,FIR,mode='valid')
    message -= message.mean()
    message /= np.sqrt(np.mean(message**2))
    signal = 1+.5*message
    assert np.allclose(clean_payload(key),signal/np.sqrt(np.mean(signal**2)))
