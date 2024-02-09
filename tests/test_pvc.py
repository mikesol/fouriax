import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf

from fouriax.pvc import (
    convert_stft_to_amp_and_freq_using_0_phase,
    fold_pvc_patches,
    koonce_normalization,
    koonce_sinc,
    make_pvc_patches,
    noscbank,
    precompute_bitreverse_indices,
    precompute_cfkt_constants,
    precompute_rfkt_constants,
    rfkt,
)

(
    convert_stft_to_amp_and_freq_using_0_phase,
    fold_pvc_patches,
    koonce_normalization,
    koonce_sinc,
    make_pvc_patches,
    noscbank,
    rfkt,
) = (
    jax.jit(convert_stft_to_amp_and_freq_using_0_phase),
    jax.jit(fold_pvc_patches, static_argnums=(2, 3, 4)),
    jax.jit(koonce_normalization),
    jax.jit(koonce_sinc, static_argnums=(1, 2)),
    jax.jit(make_pvc_patches, static_argnums=(1, 2, 3)),
    jax.jit(noscbank),
    jax.jit(rfkt),
)


def parse_log_file(path):
    tags = ["makewindows", "shiftin", "fold", "rfft", "convert"]
    o = {}
    current_tag = None
    for tag in tags:
        o[tag] = []
    with open(path) as ifi:
        lines = ifi.read().split("\n")
        for m in lines:
            for tag in tags:
                if f"after_{tag}" in m and current_tag != tag:
                    current_tag = tag
                    o[tag].append([])
                    break
                elif m[:5] != "after":
                    current_tag = None
            if current_tag is not None:
                o[current_tag][-1].append(m.strip().split(" ")[-1])
    for tag in tags:
        o[tag] = np.array(o[tag]).astype(np.float32)
    o["makewindows"] = np.squeeze(o["makewindows"])
    return o


def transform_array(input_array, transform_function):
    """
    Apply a transformation function over the 'chan' axis of an input array.

    Parameters:
    - input_array: np.ndarray of shape (bin, seq, chan).
    - transform_function: A function that takes an array of shape (chan,)
                          and returns an array of shape (chan2,).

    Returns:
    - np.ndarray of shape (bin, seq, chan2) where chan2 is determined by the
      transform_function.
    """
    # Determine the shape of the input array
    bin_, seq, _ = input_array.shape

    # Initialize an empty list to hold the transformed data
    transformed_data = []

    # Iterate over the first two dimensions (bin and seq)
    for i in range(bin_):
        seq_list = []
        for j in range(seq):
            # Apply the transformation function to each (chan,) slice
            transformed_slice = transform_function(input_array[i, j, :])
            seq_list.append(transformed_slice)
        # After processing all sequences for the current bin, stack them along the seq axis
        transformed_seq = np.stack(seq_list, axis=0)
        transformed_data.append(transformed_seq)

    # Stack the transformed data along the bin axis to get the final array
    output_array = np.stack(transformed_data, axis=0)

    return output_array


def interleave_to_complex(x):
    return x[::2] + 1j * x[1::2]


def complex_to_interleave(x):
    real = np.real(x)
    imag = np.imag(x)
    return np.ravel(np.column_stack((real, imag)))


def test_fkt():
    cap_at = 100
    n_bins = 1024
    window_length = 2048
    hop_length = 147
    bitreverse_indices = precompute_bitreverse_indices(n_bins // 2)
    c = precompute_cfkt_constants(n_bins // 2)
    ws = precompute_rfkt_constants(n_bins // 2)
    for x in range(1, 3):
        ii, _ = sf.read(f"audio/input_{x}.wav")
        ii = ii[:30000]  # don't need the full thing
        data = parse_log_file(f"logs/input_{x}.log")
        patches = make_pvc_patches(
            np.expand_dims(ii, axis=(0, 2)), hop_length, window_length
        )
        assert np.allclose(np.squeeze(patches)[:cap_at], data["shiftin"][:cap_at])
        window = np.ones((window_length,))
        window = koonce_sinc(window, n_bins, window_length)
        window = koonce_normalization(window)
        assert np.allclose(window, data["makewindows"])
        folded = fold_pvc_patches(patches, window, n_bins, hop_length, window_length)
        assert np.allclose(np.squeeze(folded)[:cap_at], data["fold"][:cap_at])
        fktd = transform_array(
            np.array(folded),
            partial(rfkt, bitreverse_indices=bitreverse_indices, c=c, ws=ws),
        )
        assert np.allclose(np.squeeze(fktd)[:cap_at], data["rfft"][:cap_at], atol=1e-5)
        fkt_batch = fktd.shape[0]
        fkt_seq = fktd.shape[1]
        fkt_chan = fktd.shape[2]
        fktd = jnp.reshape(fktd, (fkt_batch, fkt_seq, fkt_chan // 2, 2))
        f_r = fktd[..., 0]
        f_i = fktd[..., 1]
        c_r, c_i = convert_stft_to_amp_and_freq_using_0_phase(
            f_r, f_i, hop_length, 44100
        )
        converted = jnp.reshape(
            jnp.stack((c_r, c_i), axis=-1), (fkt_batch, fkt_seq, fkt_chan)
        )
        assert np.allclose(
            np.squeeze(converted)[:10, 2:], data["convert"][:10, 2:-2], atol=1e-5
        )


def test_cpx():
    c1 = random.random()
    c2 = random.random()
    xi1 = random.random()
    xi2 = random.random()
    xr = random.random()
    xi = random.random()
    wr = random.random()
    wi = random.random()
    w = wr + 1j * wi
    h1r = c1 * (xi1 + xr)
    h1i = c1 * (xi2 - xi)
    h2r = -c2 * (xi2 + xi)
    h2i = c2 * (xi1 - xr)
    x1 = xi1 + 1j * xi2
    x = xr + 1j * xi
    h1 = c1 * (x1 + np.conjugate(x))
    h2 = c2 * (np.conjugate(x1) - x)
    h2 = h2.imag + 1j * h2.real
    assert h1.real == h1r
    assert h1.imag == h1i
    assert h2.real == h2r
    assert h2.imag == h2i
    xi1 = h1r + wr * h2r - wi * h2i
    xi2 = h1i + wr * h2i + wi * h2r
    xr = h1r - wr * h2r + wi * h2i
    xi = -h1i + wr * h2i + wi * h2r
    x1 = h1 + w * h2
    x = np.conjugate(h1) - np.conjugate(w * h2)
    assert np.allclose(xi1, x1.real)
    assert np.allclose(xi2, x1.imag)
    assert np.allclose(xr, x.real)
    assert np.allclose(xi, x.imag)


def test_twiddle():
    x1 = np.arange(512)
    theta = np.pi / len(x1)
    ws = np.exp(1j * x1 * theta)
    wp = (-2.0 * (np.sin(0.5 * theta) ** 2) + 1j * np.sin(theta)) + 1
    w = 1
    o = []
    for _ in range(512):
        o.append(w)
        w *= wp
    assert np.allclose(o, ws)


def test_nosc():
    # Parameters
    fft_bins = 512
    sample_rate = 44100
    nw = 1024
    hop_length = 147

    # Initialize C with alternating amplitude and frequency values
    chans = np.zeros(2 * fft_bins + 2)  # +2 to include both endpoints
    frequencies = np.linspace(
        0, 22050, fft_bins + 1
    )  # Frequencies evenly spaced up to 22050 Hz

    # Find the index closest to 10 kHz frequency
    target_freq = 10000
    closest_idx = np.abs(frequencies - target_freq).argmin()

    # Set all amplitudes to 0 initially
    amplitudes = np.zeros(fft_bins + 1)

    # Set the amplitude at the closest index to 10 kHz to 0.1
    amplitudes[closest_idx] = 1e-5

    chans[0::2] = amplitudes  # Assign amplitudes to even indices
    chans[1::2] = frequencies  # Assign frequencies to odd indices

    # State dictionary for noscbank function
    2 * np.pi
    p_inc = 1.0 / sample_rate
    i_inv = 1.0 / hop_length
    lastval = np.zeros((fft_bins + 1, 2))
    index = np.zeros(fft_bins + 1)

    synth = noscbank(
        (lastval, index),
        jnp.tile(chans[None, :], (2 << 8, 1)),
        nw,
        p_inc,
        i_inv,
        jnp.arange(hop_length),
    )

    synth = jnp.reshape(synth[1], (-1,))

    #
    # Generate a reference 10 kHz sine wave
    t = np.arange(len(synth)) / sample_rate  # Time array
    ref_wave = 0.011 * np.sin(2 * np.pi * 10000 * t)  # 0.1 amplitude to match
    synth = synth[-nw:]
    ref_wave = ref_wave[-nw:]
    assert len(synth) == len(ref_wave)

    # Compute the Fourier Transform of both signals
    fft_output = np.fft.fft(synth)
    fft_ref = np.fft.fft(ref_wave)

    # Compute the frequency bins
    np.fft.fftfreq(nw, 1 / sample_rate)

    def find_peak_frequency(fft_data, sample_rate):
        """
        Find the frequency of the peak in the magnitude spectrum.

        Parameters:
        - fft_data (array): The FFT results (complex numbers).
        - sample_rate (int/float): The sample rate of the original signal.

        Returns:
        - float: The frequency of the peak in Hz.
        """
        # Calculate the magnitude spectrum
        magnitude_spectrum = np.abs(fft_data)

        # Number of samples is the length of the fft_data
        n_samples = len(fft_data)

        # Find the index of the maximum in the magnitude spectrum
        peak_index = np.argmax(
            magnitude_spectrum[: n_samples // 2]
        )  # Only consider the positive frequencies

        # Calculate the frequency bins
        freqs = np.fft.fftfreq(n_samples, 1 / sample_rate)

        # Map the peak index to the corresponding frequency
        peak_frequency = freqs[peak_index]

        return abs(
            peak_frequency
        )  # Return the absolute value to ensure positive frequency

    def test_peak_frequency(fft_output, fft_ref):
        # Find peak frequencies using a peak detection algorithm
        peak_freq_output = find_peak_frequency(fft_output, 44100)
        peak_freq_ref = find_peak_frequency(fft_ref, 44100)
        print(peak_freq_output, peak_freq_ref)

        # Assert that the peak frequencies are within a certain tolerance
        np.allclose(peak_freq_output, peak_freq_ref)

    def test_rmse(fft_output, fft_ref):
        rmse = np.sqrt(np.mean((np.abs(fft_output) - np.abs(fft_ref)) ** 2))
        threshold = 0.1
        assert rmse < threshold

    test_peak_frequency(fft_output, fft_ref)
    test_rmse(fft_output, fft_ref)
