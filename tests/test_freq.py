#!/usr/bin/env python

"""Tests for `fouriax` package."""

import jax
import numpy as np
import torch
from auraloss.freq import MultiResolutionSTFTLoss, STFTLoss
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.signal import butter, filtfilt

import fouriax.stft as stft
from fouriax.freq import multi_resolution_stft_loss, stft_loss

multi_resolution_stft_loss, stft_loss = jax.jit(
    multi_resolution_stft_loss,
    static_argnames=[
        "untraced_params",
        "w_sc",
        "w_log_mag",
        "w_lin_mag",
        "w_phs",
        "scale",
        "perceptual_weighting",
        "eps",
        "output",
        "reduction",
        "mag_distance",
    ],
), jax.jit(
    stft_loss,
    static_argnames=[
        "untraced_params",
        "w_sc",
        "w_log_mag",
        "w_lin_mag",
        "w_phs",
        "scale",
        "perceptual_weighting",
        "eps",
        "output",
        "reduction",
        "mag_distance",
    ],
)

shared_shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2048, max_value=4096),
        st.just(1),
    ),
    key="array_shape",
)


def filter_signal(signal):
    signal = np.array(signal)
    fs = 44100  # Sampling Frequency

    # Design Filters
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    # Apply Filters
    hp_cutoff = 1000  # High pass cutoff frequency
    lp_cutoff = 500  # Low pass cutoff frequency
    bp_low, bp_high = 200, 1500  # Bandpass frequencies

    b, a = butter_highpass(hp_cutoff, fs)
    high_passed = filtfilt(b, a, signal)

    b, a = butter_lowpass(lp_cutoff, fs)
    low_passed = filtfilt(b, a, signal)

    b, a = butter_bandpass(bp_low, bp_high, fs)
    band_passed = filtfilt(b, a, signal)

    # Mix Outputs
    mixed_output = high_passed + low_passed + band_passed
    return mixed_output


def process_batch(batch_signal):
    # Assuming batch_signal is of shape (batch, seq, chan)
    processed_signal = np.zeros_like(batch_signal)
    for i in range(batch_signal.shape[0]):  # Iterate over batch
        for j in range(batch_signal.shape[2]):  # Iterate over chan
            processed_signal[i, :, j] = filter_signal(batch_signal[i, :, j])
    return processed_signal


# we do some filtering on the hypothesis generated signals
# hypothesis has a tendency of generating stuff like all 1s, which
# leads to logs of values close to 0, which leads to numerical instability
# as no audio is actually like this, we apply filters to the generated signals
# to make them slightly more audio-y
audio_strategy = arrays(
    np.float32,
    shared_shape,
    elements={"min_value": -1.0, "max_value": 1.0},
).map(process_batch)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
    st.integers(min_value=6, max_value=8).map(lambda x: 2**x),
)
def test_stft_loss(inputs, target, res):
    """Sample pytest test function with the pytest fixture as an argument."""
    traced_params, untraced_params = stft.init_stft_params(res, res // 4, res // 2)
    loss = stft_loss(traced_params, untraced_params, inputs, target)
    loss_ref = STFTLoss(res, res // 4, res // 2)(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-3)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
)
def test_multi_resolution_stft_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_lengths = [600, 1200, 240]
    params = [
        stft.init_stft_params(x, y, z)
        for x, y, z in zip(fft_sizes, hop_sizes, win_lengths)
    ]
    traced_params, untraced_params = [[i for i, _ in params], [j for _, j in params]]
    loss = multi_resolution_stft_loss(
        traced_params, tuple(untraced_params), inputs, target
    )
    loss_ref = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref)
