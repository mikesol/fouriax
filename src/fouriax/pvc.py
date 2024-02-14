from enum import Enum
from functools import partial

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
        c = [
            jnp.concatenate([y, y * jnp.exp(1j * jnp.pi * i / len(c))])
            for i, y in enumerate(c + c)
        ]
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


def noscbank_cell(lvix, chans, nw, p_inc, i_inv, rg):
    """
    Generates a bank of oscillators with varying amplitudes and frequencies over a specified range,
    accumulating their outputs. This function is designed to process and synthesize audio signals
    based on a set of input control parameters for amplitude and frequency modulation.

    Parameters:
    - lvix (tuple): A tuple containing two elements:
        - lv (array): The last value of amplitude and frequency for each oscillator.
        - ix (array): The current index or phase of each oscillator.
    - C (array): A 2D array of control parameters where each row contains an amplitude and frequency
      value for an oscillator. The array shape should be (-1, 2), where each row represents [amplitude, frequency].
    - Nw (int): The number of samples in the window, used for scaling the oscillator outputs.
    - Pinc (float): The phase increment per sample, derived from the sampling rate.
    - Iinv (float): The inverse of the interpolation factor, used for controlling the rate of change in amplitude and frequency.
    - rg (array): An array of indices used for generating the oscillator outputs over time.

    Returns:
    - A tuple containing:
        - A tuple with the last values of amplitude and frequency for each oscillator, and the final index or phase.
        - The accumulated output of all oscillators, summed over the second axis.

    This function utilizes JAX for efficient numerical computations and parallel processing of oscillators.
    The internal mechanism involves mapping a function over each oscillator's control parameters, followed
    by a loop to generate the oscillator's output using cosine modulation based on the current phase and frequency.
    The outputs are then summed to produce the final signal.

    Example usage:
    ```
    # Assuming lvix, C, Nw, Pinc, Iinv, and rg are defined as per the function requirements.
    last_values, index, output_signal = noscbank_cell(lvix, C, Nw, Pinc, Iinv, rg)
    ```
    """
    lv, ix = lvix

    def _vmap(c, lastval, index):

        finc = (c[1] * p_inc - lastval[1]) * i_inv
        ainc = (c[0] - lastval[0]) * i_inv
        address = index

        def _loop(carry, _):
            addr, lastval = carry
            idx = addr
            o_chan = lastval[0] * nw * jnp.cos(TWOPI * idx)
            addr += lastval[1]

            lastval += jnp.array([ainc, finc])
            return (addr, lastval), o_chan

        carry, o = jax.lax.scan(_loop, (address, lastval), rg)
        address = carry[0]
        lastval = jnp.array([c[0], c[1] * p_inc])
        index = address
        return o, lastval, index

    chans = jnp.reshape(chans, (-1, 2))
    o, lastval, index = jax.vmap(_vmap, in_axes=0, out_axes=0)(chans, lv, ix)
    return (lastval, index), jnp.sum(o, axis=0)


def noscbank(lvix, chans, nw, p_inc, i_inv, rg):
    return jax.lax.scan(
        partial(noscbank_cell, nw=nw, p_inc=p_inc, i_inv=i_inv, rg=rg), lvix, chans
    )


# expect i to be a 1-d vector alternating amp, freq
def _phaselock(freqs):
    freqs = jnp.reshape(freqs, (1, 1, -1))
    _tf = jax.lax.conv_general_dilated_patches(
        freqs,
        filter_shape=(3,),
        window_strides=(1,),
        padding=((1, 1),),
    )
    _tf = jnp.reshape(_tf, (3, -1))
    print("tf shape", _tf.shape, _tf.dtype)
    tf = jax.vmap(
        lambda x: jnp.where((x[1] > x[0]) & (x[1] > x[2]), 1.0, -1.0),
        in_axes=-1,
        out_axes=-1,
    )(_tf)
    print("shaaape", tf.shape[-1], freqs.shape[-1])
    assert tf.shape[-1] == freqs.shape[-1]

    def select_y_element(x, y):
        cond1 = (x[0] < 0.0) & (x[2] < 0.0)
        cond2 = (x[0] > 0.0) & (x[2] <= 0.0)
        cond3 = (x[2] > 0.0) & (x[0] <= 0.0)
        cond4 = x[0] == 0
        cond5 = x[2] == 0
        cond6 = y[0] > y[2]

        return jnp.where(
            cond1,
            y[1],
            jnp.where(
                cond2,
                y[0],
                jnp.where(
                    cond3,
                    y[2],
                    jnp.where(
                        cond4,
                        y[1],
                        jnp.where(cond5, y[1], jnp.where(cond6, y[0], y[2])),
                    ),
                ),
            ),
        )

    tf = jnp.reshape(tf, (1, 1, -1))
    tf = jax.lax.conv_general_dilated_patches(
        tf,
        filter_shape=(3,),
        window_strides=(1,),
        padding=((1, 1),),
    )
    tf = jnp.reshape(tf, (3, -1))
    o = jax.vmap(
        select_y_element,
        in_axes=-1,
        out_axes=-1,
    )(tf, _tf)
    assert o.shape[-1] == freqs.shape[-1]
    return o


def phaselock(i):
    freqs = i[1::2]
    o = _phaselock(freqs)
    return jnp.ravel(jnp.column_stack((i[::2], o)))
