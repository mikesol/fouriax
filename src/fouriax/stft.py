import dataclasses
from typing import Optional

import jax.lax as lax
import jax.numpy as jnp
import librosa


@dataclasses.dataclass(frozen=True)
class DftParams:
    n: int
    norm: Optional[str]


@dataclasses.dataclass(frozen=True)
class StftParams:
    hop_length: int
    n_fft: int


def init_dft_params(n, norm=None):
    """
    Initialize the DFT and IDFT matrices (parameters) for JAX operations.

    Args:
    n (int): Size of the DFT and IDFT matrices.
    norm (str or None): Normalization mode, either None or 'ortho'.

    Returns:
    A dictionary containing DFT and IDFT matrices as well as other parameters.
    """

    def dft_matrix(n):
        x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
        omega = jnp.exp(-2 * jnp.pi * 1j / n)
        w = omega ** (x * y)
        return w

    def idft_matrix(n):
        x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
        omega = jnp.exp(2 * jnp.pi * 1j / n)
        w = omega ** (x * y)
        return w

    w = dft_matrix(n)
    inv_w = idft_matrix(n)

    return {
        "W_real": jnp.real(w),
        "W_imag": jnp.imag(w),
        "inv_W_real": jnp.real(inv_w),
        "inv_W_imag": jnp.imag(inv_w),
    }, DftParams(n, norm)


def dft(traced_params, untraced_params, x_real, x_imag):
    """
    Perform the Discrete Fourier Transform (DFT) using JAX.

    Args:
    params (dict): Parameters containing the DFT matrices.
    x_real (array): Real part of the signal.
    x_imag (array): Imaginary part of the signal.

    Returns:
    Tuple of arrays representing the real and imaginary parts of the DFT.
    """
    z_real = jnp.dot(x_real, traced_params["W_real"]) - jnp.dot(
        x_imag, traced_params["W_imag"]
    )
    z_imag = jnp.dot(x_imag, traced_params["W_real"]) + jnp.dot(
        x_real, traced_params["W_imag"]
    )

    if untraced_params.norm == "ortho":
        z_real /= jnp.sqrt(untraced_params.n)
        z_imag /= jnp.sqrt(untraced_params.n)

    return z_real, z_imag


def idft(traced_params, untraced_params, x_real, x_imag):
    """
    Perform the Inverse Discrete Fourier Transform (IDFT) using JAX.

    Args:
    params (dict): Parameters containing the IDFT matrices.
    x_real (array): Real part of the signal.
    x_imag (array): Imaginary part of the signal.

    Returns:
    Tuple of arrays representing the real and imaginary parts of the IDFT.
    """
    z_real = jnp.dot(x_real, traced_params["inv_W_real"]) - jnp.dot(
        x_imag, traced_params["inv_W_imag"]
    )
    z_imag = jnp.dot(x_imag, traced_params["inv_W_real"]) + jnp.dot(
        x_real, traced_params["inv_W_imag"]
    )

    if untraced_params.norm is None:
        z_real /= untraced_params.n
    elif untraced_params.norm == "ortho":
        z_real /= jnp.sqrt(untraced_params.n)
        z_imag /= jnp.sqrt(untraced_params.n)

    return z_real, z_imag


def rdft(traced_params, untraced_params, x_real):
    """
    Perform the Right-side Real Discrete Fourier Transform (RDFT) using JAX.

    Args:
    traced_params (dict): Traced parameters containing the DFT matrices.
    untraced_params (dict): Untraced static parameters.
    x_real (array): Real part of the signal.

    Returns:
    Tuple of arrays representing the real and imaginary parts of the RDFT.
    """
    n_rfft = untraced_params.n // 2 + 1
    z_real = jnp.dot(x_real, traced_params["W_real"][..., :n_rfft])
    z_imag = jnp.dot(x_real, traced_params["W_imag"][..., :n_rfft])

    if untraced_params.norm == "ortho":
        z_real /= jnp.sqrt(untraced_params.n)
        z_imag /= jnp.sqrt(untraced_params.n)

    return z_real, z_imag


def irdft(traced_params, untraced_params, x_real, x_imag):
    """
    Perform the Inverse Real Discrete Fourier Transform (IRDFT) using JAX.

    Args:
    traced_params (dict): Traced parameters containing the IDFT matrices.
    untraced_params (dict): Untraced static parameters.
    x_real (array): Real part of the signal (n // 2 + 1,).
    x_imag (array): Imaginary part of the signal (n // 2 + 1,).

    Returns:
    An array representing the real part of the output signal.
    """
    n_rfft = untraced_params.n // 2 + 1

    # Flip and concatenate to reconstruct full signal
    flip_x_real = jnp.flip(x_real, axis=-1)
    flip_x_imag = jnp.flip(x_imag, axis=-1)

    x_real_full = jnp.concatenate((x_real, flip_x_real[..., 1 : n_rfft - 1]), axis=-1)
    x_imag_full = jnp.concatenate(
        (x_imag, -1.0 * flip_x_imag[..., 1 : n_rfft - 1]), axis=-1
    )

    # Calculate IRDFT
    z_real = jnp.dot(x_real_full, traced_params["inv_W_real"]) - jnp.dot(
        x_imag_full, traced_params["inv_W_imag"]
    )

    if untraced_params.norm == "ortho":
        z_real /= jnp.sqrt(untraced_params.n)

    return z_real


# from librosa
def pad_center(data, *, size, axis: int = -1):
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")

    return jnp.pad(data, lengths)


def init_stft_params(n_fft, hop_length=None, win_length=None, window_type="hann"):
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    # Window function
    fft_window = jnp.array(librosa.filters.get_window(window_type, win_length))
    fft_window = pad_center(fft_window, size=n_fft)

    # DFT matrix
    def dft_matrix(n):
        x, y = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
        omega = jnp.exp(-2 * jnp.pi * 1j / n)
        w = omega ** (x * y)
        return w

    w = dft_matrix(
        n_fft
    )  # Reuse the dft_matrix function from the previous implementation

    # Prepare convolutional filters
    real_filter = jnp.real(w[:, : n_fft // 2 + 1] * fft_window[:, None]).T
    imag_filter = jnp.imag(w[:, : n_fft // 2 + 1] * fft_window[:, None]).T

    return {
        "real_filter": real_filter[None, :],
        "imag_filter": imag_filter[None, :],
    }, StftParams(hop_length, n_fft)


def stft(traced_params, untraced_params, input_signal, center=True, pad_mode="reflect"):
    batch_size = input_signal.shape[0]
    n_fft = untraced_params.n_fft
    hop_length = untraced_params.hop_length

    if center:
        input_signal = jnp.pad(
            input_signal, ((0, 0), (n_fft // 2, n_fft // 2)), mode=pad_mode
        )

    # Convolution parameters
    stride = (hop_length,)

    # Real and Imaginary convolutions

    real = lax.conv_general_dilated(
        input_signal[:, None, :],
        traced_params["real_filter"],
        window_strides=stride,
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=("NCH", "IOH", "NCH"),
        feature_group_count=1,
    )

    imag = lax.conv_general_dilated(
        input_signal[:, None, :],
        traced_params["imag_filter"],
        window_strides=stride,
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=("NCH", "IOH", "NCH"),
        feature_group_count=1,
    )

    def last_reshape(x):
        x = jnp.reshape(x, (-1, batch_size, x.shape[1], x.shape[2]))
        x = jnp.sum(x, axis=0)
        x = jnp.transpose(x, (0, 2, 1))
        return x

    real, imag = last_reshape(real), last_reshape(imag)
    return real, imag
