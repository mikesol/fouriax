#!/usr/bin/env python

"""Tests for `fouriax` package."""

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays, from_dtype
import torch
import fouriax.stft as stft

from fouriax.freq import *
from auraloss.freq import *

shared_shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2048, max_value=4096),
        st.just(1),
    ),
    key="array_shape",
)

audio_strategy = arrays(
    np.float32, shared_shape, elements=dict(min_value=-1.0, max_value=1.0)
)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
    st.integers(min_value=6, max_value=8).map(lambda x: 2**x),
)
def test_stft_loss(input, target, res):
    """Sample pytest test function with the pytest fixture as an argument."""
    params = stft.init_stft_params(res, res // 4, res // 2)
    loss = stft_loss(params, input, target)
    loss_ref = STFTLoss(res, res // 4, res // 2)(
        torch.from_numpy(np.transpose(input, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
)
def test_multi_resolution_stft_loss(input, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_lengths = [600, 1200, 240]
    params = [
        stft.init_stft_params(x, y, z)
        for x, y, z in zip(fft_sizes, hop_sizes, win_lengths)
    ]
    loss = multi_resolution_stft_loss(params, input, target)
    loss_ref = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )(
        torch.from_numpy(np.transpose(input, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref)
