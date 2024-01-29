import jax.numpy as jnp
import scipy.signal
from jax import lax


def create_fir_filter(filter_type, coef, fs, ntaps):
    if ntaps % 2 == 0:
        raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

    if filter_type == "hp":
        taps = jnp.array([1, -coef, 0])
    elif filter_type == "fd":
        taps = jnp.array([1, 0, -coef])
    elif filter_type == "aw":
        f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
        a1000 = 1.9997
        nums = [(2 * jnp.pi * f4) ** 2 * (10 ** (a1000 / 20)), 0, 0, 0, 0]
        dens = jnp.polymul(
            jnp.array([1, 4 * jnp.pi * f4, (2 * jnp.pi * f4) ** 2]),
            jnp.array([1, 4 * jnp.pi * f1, (2 * jnp.pi * f1) ** 2]),
        )
        dens = jnp.polymul(
            jnp.polymul(dens, jnp.array([1, 2 * jnp.pi * f3])),
            jnp.array([1, 2 * jnp.pi * f2]),
        )
        b, a = scipy.signal.bilinear(nums, dens, fs=fs)
        w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)
        taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")

    return taps


def fir_filter(signal, kernel, ntaps):
    # Reshape filter kernel for convolution
    padding = ntaps // 2
    kernel = kernel.reshape((1, 1, -1))

    # Apply FIR filter using 1D convolution
    filtered_signal = lax.conv_general_dilated(
        signal[:, None, :],  # Add channel dimension
        kernel,
        window_strides=(1,),
        padding=((padding, padding),),
        dimension_numbers=("NCH", "IOH", "NCH"),
        feature_group_count=1,
    )

    return filtered_signal.squeeze()  # Remove channel dimension
