from enum import Enum

import jax
import jax.numpy as jnp


class PVCPatchPadding(Enum):
    ZERO = 1
    BLEED = 2


def precompute_bitreverse_indices(n):
    def _alg(i):
        m = i * 2
        return jnp.concatenate([m, m + 1])

    o = jnp.array([0])
    while len(o) < n:
        o = _alg(o)
    return o


def precompute_cfkt_constants(n):
    c = [jnp.array([1])]
    while len(c) < n:
        cc = c + c
        t = []
        for i, y in enumerate(cc):
            t.append(jnp.concatenate([y, y * jnp.exp(1j * jnp.pi * i / len(c))]))
        c = t
    c = jnp.array(c)
    return c


def precompute_rfkt_constants(n):
    theta = jnp.pi / n
    first_half = jnp.arange((n >> 1) + 1)
    ws = jnp.exp(1j * first_half * theta)
    return ws


def make_pvc_patches(i, hop_length, window_length, padding=PVCPatchPadding.BLEED):
    assert len(i.shape) == 3
    batch_size = i.shape[0]
    o = jax.lax.conv_general_dilated_patches(
        jnp.transpose(i, (0, 2, 1)),
        filter_shape=(window_length,),
        window_strides=(hop_length,),
        padding=(
            ((window_length - hop_length, window_length - hop_length),)
            if padding == PVCPatchPadding.BLEED
            else ((0, 0),)
        ),
    )
    return jnp.transpose(
        jnp.reshape(o, (batch_size, window_length, -1)),
        (0, 2, 1),
    )


def cfkt(xc, bitreverse_indices, c):
    nd = len(xc) * 2
    xc = xc[bitreverse_indices]
    xc = jnp.dot(c, xc)
    # Scale output
    scale = 1.0 / nd
    xc *= scale
    return xc


PI = jnp.pi
TWOPI = 2.0 * jnp.pi


def rfkt(xx, bitreverse_indices, c, ws):
    n = len(xx) // 2
    x = xx[::2] + 1j * xx[1::2]
    x = cfkt(x, bitreverse_indices, c)
    c1 = 0.5
    c2 = -c1
    x = jnp.concatenate([x, jnp.array([x[0]])])
    first_half = jnp.arange((n >> 1) + 1)
    second_half = -first_half - 1  # Mirrored indices
    x1 = x[first_half]
    x3 = x[second_half]
    h1 = c1 * (x1 + jnp.conjugate(x3))
    h2 = c2 * (jnp.conjugate(x1) - x3)
    h2 = h2.imag + 1j * h2.real
    x = jnp.concatenate(
        [(h1 + ws * h2)[:-1], jnp.flip(jnp.conjugate(h1) - jnp.conjugate(ws * h2))]
    )
    x = jnp.concatenate([jnp.array([x[0].real + 1j * x[-1].real]), x[1:]])
    x = x[:-1]
    real = jnp.real(x)
    imag = jnp.imag(x)
    x_final = jnp.ravel(jnp.column_stack((real, imag)))

    return x_final


def koonce_sinc(i, n, nw):
    x = (-(nw - 1) / 2.0) + jnp.arange(nw)
    pix = jnp.pi * x
    return i * n * jnp.sin(pix / n) / pix


def koonce_normalization(x):
    return x * 2.0 / jnp.sum(x)


def adjust_values(n, n_bins):
    adjustment = jnp.where(n < 0, ((-n - 1) // n_bins + 1) * n_bins, 0)
    n_adjusted = n + adjustment
    n_final = n_adjusted % n_bins
    return n_final


def fold_pvc_patches(x, w, n_bins, hop_length, window_length):
    assert len(x.shape) == 3
    batch_size = x.shape[0]
    seq = x.shape[1]
    seq_range = jnp.arange(seq) * hop_length - (window_length - hop_length)
    seq_range = jnp.expand_dims(
        window_length - adjust_values(seq_range, n_bins), axis=(0, -1)
    )
    m = x * jnp.expand_dims(w, axis=(0, 1))
    win_range = jnp.arange(window_length)
    m = jnp.concatenate([m, m], axis=-1)
    m = jax.vmap(lambda x, y: x[:, jnp.squeeze(y) + win_range], in_axes=1, out_axes=1)(
        m, seq_range
    )
    padding = 0 if window_length % n_bins == 0 else n_bins - (window_length % n_bins)
    padding_config = [(0, 0) for _ in range(m.ndim - 1)] + [(0, padding)]
    m = jnp.pad(m, padding_config)
    m = jnp.reshape(m, (batch_size, seq, -1, n_bins))
    m = jnp.sum(m, axis=2)
    return m


TWOPI = 2 * jnp.pi


def convert_dft_to_amp_and_freq(real, imag, lastphase, d, r):
    n_bins = real.shape[-1]
    fft_len = n_bins * 2

    # Compute constants
    fundamental = r / fft_len
    factor = r / (d * TWOPI)

    # Compute magnitude (amplitude)
    amplitude = jnp.hypot(real, imag)

    # Compute phase with negation as in the original C code: -atan2(imag, real)
    phase = -jnp.arctan2(imag, real)

    # Compute phase difference as in the C code
    phasediff = phase - lastphase

    # Phase unwrapping, ensuring it matches the logic in the C code
    phasediff = (phasediff + jnp.pi) % TWOPI - jnp.pi

    # Frequency calculation
    frequency = (
        phasediff * factor
        + jnp.expand_dims(jnp.arange(n_bins), axis=(0,)) * fundamental
    )

    # Update lastphase for next call, using the computed phase
    lastphase_updated = phase

    return lastphase_updated, jnp.stack([amplitude, frequency], axis=-1)


def convert_stft_to_amp_and_freq(real, imag, lastphase, d, r):
    stacked_ri = jnp.stack([real, imag], axis=-1)
    _, o = jax.lax.scan(
        lambda c, x: convert_dft_to_amp_and_freq(x[..., 0], x[..., 1], c, d, r),
        lastphase,
        jnp.transpose(stacked_ri, (1, 0, 2, 3)),
    )
    o = jnp.transpose(o, (1, 0, 2, 3))
    return o[..., 0], o[..., 1]


def convert_stft_to_amp_and_freq_using_0_phase(real, imag, d, r):
    return convert_stft_to_amp_and_freq(real, imag, jnp.zeros_like(real[:, 0, :]), d, r)
