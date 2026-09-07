"""Versioned per-frame streams, mathematical IQ controls and observation-only LS.

Complex reference arithmetic is float64; only the final model NCT input is float32.
Neither pilot_estimate nor channel_observation accepts true SNR.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import firwin

VERSION = 1
SEED = 20260906
CLASSES = ('BPSK', 'QPSK', '8PSK', '16QAM', 'DSB_mathematical', 'FM_mathematical')
COUNTS = {'train': 600, 'validation': 200, 'test': 200}
PARTITIONS = {'train': 0, 'validation': 1, 'test': 2}
CHANNELS = {'clean': 0, 'A': 1, 'B': 2}
PURPOSES = {'symbols': 0, 'offset': 1, 'message': 2, 'channel': 3,
            'cfo': 4, 'gain': 5, 'payload_noise': 6, 'pilot_noise': 7}
FIR = firwin(65, .05, window='hamming', scale=True).astype(np.float64)
FIR.setflags(write=False)


@dataclass(frozen=True)
class FrameID:
    partition: str
    class_index: int
    snr_index: int
    frame_index: int

    def __post_init__(self):
        if (self.partition not in COUNTS or
            any(type(v) is not int for v in (self.class_index, self.snr_index, self.frame_index)) or
            not 0 <= self.class_index < 6 or not 0 <= self.snr_index < 20 or
            not 0 <= self.frame_index < COUNTS[self.partition]):
            raise ValueError('invalid fixed partition/cell/frame identity')

    @property
    def name(self):
        return f'p12v{VERSION}:{self.partition}:c{self.class_index}:s{self.snr_index}:f{self.frame_index}'

    def stream_key(self, channel, purpose):
        return [VERSION, SEED, PARTITIONS[self.partition], self.class_index,
                self.snr_index, self.frame_index, CHANNELS[channel], PURPOSES[purpose]]

    def rng(self, channel, purpose):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.stream_key(channel, purpose))))


def constellation(label):
    if label in (0, 1, 2):
        count = (2, 4, 8)[label]
        return np.exp(2j*np.pi*np.arange(count)/count)
    if label == 3:
        axis = np.array([-3., -1., 1., 3.])
        return (axis[:, None]+1j*axis[None, :]).ravel()/np.sqrt(10.)
    raise ValueError('constellation only defined for four digital classes')


def clean_payload(key):
    if key.class_index < 4:
        points = constellation(key.class_index)
        symbols = points[key.rng('clean', 'symbols').integers(len(points), size=34)]
        offset = int(key.rng('clean', 'offset').integers(4))
        # 34 contextual symbols; crop 128 samples at 4+offset without padding.
        clean = np.repeat(symbols, 4)[4+offset:132+offset]
    else:
        message = np.convolve(key.rng('clean', 'message').standard_normal(192), FIR, mode='valid')
        message -= message.mean()
        message /= np.sqrt(np.mean(message**2))
        clean = (1.+.5*message).astype(complex) if key.class_index == 4 else np.exp(.35j*np.cumsum(message))
        # FM phase at the sample before the crop is zero; channel adds random phase.
    return np.asarray(clean/np.sqrt(np.mean(abs(clean)**2)), dtype=np.complex128)


def proper_noise(rng, shape, variance):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('noise variance must be finite and nonnegative')
    return np.sqrt(variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))


def pilot_sequence(k=64):
    if k not in (8, 16, 32, 64):
        raise ValueError('pilot sequence length outside registered positive K')
    return np.where(np.arange(-k, 0) % 2 == 0, 1., -1.).astype(np.complex128)


def channel_observation(clean, times, coefficient, cfo, gain, noise):
    clean, times, noise = np.asarray(clean), np.asarray(times), np.asarray(noise)
    if clean.shape != times.shape or clean.shape != noise.shape:
        raise ValueError('unaligned clean/time/noise arrays')
    return gain*(coefficient*clean*np.exp(2j*np.pi*cfo*times)+noise)


def generate_frame(key, channel='A'):
    if channel not in ('A', 'B') or (channel == 'B' and key.partition != 'test'):
        raise ValueError('unseen Channel B is test-only')
    clean = clean_payload(key)
    rng = key.rng(channel, 'channel')
    h = np.exp(1j*rng.uniform(-np.pi, np.pi)) if channel == 'A' else complex(*rng.standard_normal(2))/np.sqrt(2)
    cfo = 0. if channel == 'A' else key.rng(channel, 'cfo').uniform(-.02, .02)
    # Shared independent receiver gain for the paired A/B frame isolates channel changes.
    gain_db = key.rng('clean', 'gain').uniform(-6., 6.)
    gain = 10**(gain_db/20.)
    target_db = -20.+2*key.snr_index
    post_channel_power = float(np.mean(abs(h*clean)**2))
    variance = post_channel_power / 10**(target_db/10.)
    noise = proper_noise(key.rng(channel, 'payload_noise'), (128,), variance)
    pilot_noise = proper_noise(key.rng(channel, 'pilot_noise'), (64,), variance)
    payload = channel_observation(clean, np.arange(128), h, cfo, gain, noise)
    pilot = channel_observation(pilot_sequence(), np.arange(-64,0), h, cfo, gain, pilot_noise)
    keys = {purpose: key.stream_key('clean' if purpose in ('symbols','offset','message','gain') else channel, purpose) for purpose in PURPOSES}
    return dict(waveform_id=key.name, channel=channel, class_index=key.class_index,
                target_snr_db=target_db, clean=clean, payload=payload, pilot=pilot,
                payload_noise=noise, pilot_noise=pilot_noise, stream_keys=keys,
                channel_coefficient=h, cfo=float(cfo), gain_db=float(gain_db), gain_amplitude=float(gain),
                clean_power=float(np.mean(abs(clean)**2)), post_channel_power=post_channel_power,
                noise_variance_before_gain=variance, noise_variance_after_gain=variance*gain**2,
                received_expected_signal_power=post_channel_power*gain**2,
                observed_power=float(np.mean(abs(payload)**2)),
                nct=np.stack((payload.real, payload.imag)).astype(np.float32))


def pilot_estimate(observed, known, *, clip=(-20., 18.), epsilon=1e-12):
    """Last dimension contains pilots. No target/channel truth or fit state allowed."""
    y, p = np.asarray(observed, dtype=np.complex128), np.asarray(known, dtype=np.complex128)
    if y.ndim < 1 or p.ndim != 1 or y.shape[-1] != len(p):
        raise ValueError('unaligned observed and known pilots')
    k = len(p)
    if k == 0:
        raise ValueError('K=0 requires the separately trained frame-estimator fallback')
    if k not in (8,16,32,64) or not np.isfinite(y).all() or not np.isfinite(p).all() or not np.allclose(abs(p), 1., rtol=0, atol=1e-14):
        raise ValueError('finite unit-modulus registered pilots required')
    if not np.isfinite(epsilon) or epsilon <= 0 or len(clip) != 2 or not np.isfinite(clip).all() or clip[0] >= clip[1]:
        raise ValueError('invalid predeclared numerical guards')
    energy = np.sum(abs(p)**2)
    h = np.sum(np.conj(p)*y, axis=-1)/energy
    noise = np.sum(abs(y-h[..., None]*p)**2, axis=-1)/(k-1)
    signal = abs(h)**2-noise/energy
    signed = np.full(np.shape(signal), np.nan)
    np.divide(signal, noise, out=signed, where=noise > 0)
    valid = (signal > 0) & (noise > 0)
    raw = np.full(np.shape(signal), np.nan)
    np.log10(signed, out=raw, where=valid)
    raw *= 10
    guarded = 10*np.log10(np.maximum(signal,epsilon)/np.maximum(noise,epsilon))
    return dict(pilot_count=k, coefficient=h, pilot_energy=float(energy), noise_variance=noise,
                corrected_signal_power=signal, signed_linear_snr=signed, raw_db_valid=valid,
                raw_db=raw, guarded_db=guarded, clipped_db=np.clip(guarded, *clip),
                zero_noise_guard=noise <= 0, nonpositive_signal_guard=signal <= 0,
                noise_floor_guard=noise < epsilon, signal_floor_guard=signal < epsilon)


def overhead(k, accuracy):
    if k not in (0,8,16,32,64) or not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError('invalid registered overhead/accuracy')
    fraction = 128/(128+k)
    return dict(payload_fraction=fraction, overhead=k/(128+k), utility_proxy=fraction*accuracy)
