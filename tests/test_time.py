#!/usr/bin/env python

"""Tests for `fouriax` package."""

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays, from_dtype
import torch
import jax

from fouriax.time import *
from auraloss.time import *

shared_shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=16, max_value=20247),
        st.just(1),
    ),
    key="array_shape",
)

audio_strategy = arrays(
    np.float32, shared_shape, elements=dict(min_value=-1.0, max_value=1.0)
)


@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_log_cosh_loss(input, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(log_cosh_loss)(input, target, eps=1e-5),
        LogCoshLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(input, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )


@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_esr_loss(input, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(esr_loss)(input, target, eps=1e-5),
        ESRLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(input, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )

@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_dc_loss(input, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(dc_loss)(input, target, eps=1e-5),
        DCLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(input, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )
