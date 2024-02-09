import jax
import numpy as np
import torch
from auraloss.perceptual import FIRFilter
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fouriax.perceptual import create_fir_filter, fir_filter

fir_filter = jax.jit(fir_filter, static_argnums=(2,))
audio_strategy = arrays(
    np.float32,
    st.tuples(
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=16, max_value=20247),
    ),
    elements={"min_value": -1.0, "max_value": 1.0},
)


@settings(deadline=None)
@given(audio_strategy)
def test_fir_filter_hp(signal):
    taps = create_fir_filter(filter_type="hp", coef=0.85, fs=44100, ntaps=101)
    filtered_signal = fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = FIRFilter(
        filter_type="hp", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )


@settings(deadline=None)
@given(audio_strategy)
def test_fir_filter_fd(signal):
    taps = create_fir_filter(filter_type="fd", coef=0.85, fs=44100, ntaps=101)
    filtered_signal = fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = FIRFilter(
        filter_type="fd", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )


@settings(deadline=None)
@given(audio_strategy)
def test_fir_filter_aw(signal):
    taps = create_fir_filter(filter_type="aw", coef=0.85, fs=44100, ntaps=101)
    filtered_signal = fir_filter(signal, taps, 101)
    torch_sig = torch.from_numpy(np.expand_dims(signal, axis=-2))
    filtered_signal_ref, _ = FIRFilter(
        filter_type="aw", coef=0.85, fs=44100, ntaps=101
    )(torch_sig, torch_sig)
    assert np.allclose(
        filtered_signal, np.squeeze(np.array(filtered_signal_ref)), atol=1.0e-3
    )
