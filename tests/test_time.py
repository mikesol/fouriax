#!/usr/bin/env python

"""Tests for `fouriax` package."""

import jax
import numpy as np
import torch
from auraloss.time import DCLoss, ESRLoss, LogCoshLoss
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fouriax.time import dc_loss, esr_loss, log_cosh_loss

dc_loss, esr_loss, log_cosh_loss = (
    jax.jit(dc_loss, static_argnums=(2,)),
    jax.jit(esr_loss, static_argnums=(2,)),
    jax.jit(log_cosh_loss, static_argnums=(2, 3)),
)

shared_shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=16, max_value=20247),
        st.just(1),
    ),
    key="array_shape",
)

audio_strategy = arrays(
    np.float32,
    shared_shape,
    elements={"min_value": -1.0, "max_value": 1.0},
)


@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_log_cosh_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(log_cosh_loss)(inputs, target, eps=1e-5),
        LogCoshLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )


@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_esr_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(esr_loss)(inputs, target, eps=1e-5),
        ESRLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )


@settings(deadline=None)
@given(audio_strategy, audio_strategy)
def test_dc_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert np.allclose(
        jax.jit(dc_loss)(inputs, target, eps=1e-5),
        DCLoss(eps=1e-5)(
            torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
            torch.from_numpy(np.transpose(target, (0, 2, 1))),
        ),
        atol=1e-3,
    )
