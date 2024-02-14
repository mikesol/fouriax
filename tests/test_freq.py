#!/usr/bin/env python

"""Tests for `fouriax` package."""

import jax
import numpy as np
import torch
from auraloss.freq import MultiResolutionSTFTLoss, STFTLoss
from hypothesis import given, settings
from hypothesis import strategies as st

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


fs = 44100  # Sampling rate


@st.composite
def generate_sine_wave(draw, length):
    """Composite strategy to generate a single sine wave."""
    amplitude = draw(st.floats(min_value=0.01, max_value=1.0))
    frequency = draw(st.floats(min_value=30.0, max_value=22050.0))
    t = np.linspace(0, length / fs, int(length), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave


@st.composite
def generate_complex_signal(draw, shape):
    """Composite strategy to generate a complex signal from multiple sine waves."""
    length = shape[1]  # Using the second element of shape for signal length
    sine_waves = draw(st.lists(generate_sine_wave(length), min_size=32, max_size=32))
    complex_signal = np.sum(sine_waves, axis=0)
    # Normalize the complex signal to ensure it's within [-1, 1]
    max_amplitude = np.max(np.abs(complex_signal))
    if max_amplitude > 0:  # Avoid division by zero
        complex_signal /= max_amplitude
    return complex_signal.reshape(shape)


# Define the audio strategy
audio_strategy = generate_complex_signal((1, fs, 1))


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
    assert np.allclose(loss, loss_ref, atol=1.0e-1)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
)
def test_multi_resolution_stft_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    fft_sizes = [256, 512]
    hop_sizes = [64, 128]
    win_lengths = [128, 256]
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
    assert np.allclose(loss, loss_ref, atol=1.0e-1)
