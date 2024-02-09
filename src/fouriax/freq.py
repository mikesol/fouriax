from functools import partial

import jax
import jax.numpy as jnp
import optax

from . import stft


def spectral_convergence_loss(x_mag, y_mag):
    """
    Calculate the spectral convergence loss.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.

    Returns:
        The spectral convergence loss.
    """
    numerator = jnp.linalg.norm(y_mag - x_mag, ord=2)
    denominator = jnp.linalg.norm(y_mag, ord=2)
    loss = numerator / denominator
    return loss


def stft_magnitude_loss(x_mag, y_mag, log=True, distance="L1", reduction="mean"):
    """
    Calculate the STFT magnitude loss.

    Args:
        x_mag (array): The magnitude spectrum of the first signal.
        y_mag (array): The magnitude spectrum of the second signal.
        log (bool): Whether to log-scale the STFT magnitudes.
        distance (str): Distance function ["L1", "L2"].
        reduction (str): Reduction of the loss elements ["mean", "sum", "none"].

    Returns:
        The STFT magnitude loss.
    """
    if log:
        x_mag = jnp.log(x_mag + 1e-8)  # Adding a small value to avoid log(0)
        y_mag = jnp.log(y_mag + 1e-8)

    if distance == "L1":
        # L1 Loss (Mean Absolute Error)
        loss = jnp.abs(x_mag - y_mag)
    elif distance == "L2":
        # L2 Loss (Mean Squared Error)
        loss = (x_mag - y_mag) ** 2
    else:
        raise ValueError(f"Invalid distance: '{distance}'.")

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: '{reduction}'.")


def stft_loss(
    traced_params,
    untraced_params,
    inputs,
    target,
    w_sc=1.0,
    w_log_mag=1.0,
    w_lin_mag=0.0,
    w_phs=0.0,
    scale=None,
    perceptual_weighting=None,
    eps=1e-8,
    output="loss",
    reduction="mean",
    mag_distance="L1",
):
    """
    Calculate the STFT loss.
    """
    bs, _, chs = inputs.shape

    # Compute STFT for inputs and target
    def _to_map(x):
        o = stft.stft(traced_params, untraced_params, x)
        return o

    to_map = jax.vmap(_to_map, in_axes=-1, out_axes=-1)

    def mf(x, y):
        return (
            jnp.sqrt(jnp.clip((x**2) + (y**2), a_min=eps)),
            jax.lax.atan2(y, x),
        )

    inputs_mag, inputs_phs = mf(*to_map(inputs))
    target_mag, target_phs = mf(*to_map(target))

    # Apply scaling (e.g., Mel, Chroma) if required
    if scale is not None:
        inputs_mag = jnp.matmul(scale, inputs_mag)
        target_mag = jnp.matmul(scale, target_mag)

    # Apply perceptual weighting if required
    if perceptual_weighting is not None:
        # since FIRFilter only support mono audio we will move channels to batch dim
        inputs = jnp.transpose(inputs, (0, 2, 1))
        target = jnp.transpose(target, (0, 2, 1))
        inputs = jnp.reshape(inputs, (bs * chs, -1))
        target = jnp.reshape(target, (bs * chs, -1))

        # now apply the filter to both
        inputs, target = perceptual_weighting(inputs), perceptual_weighting(target)

        # now move the channels back
        inputs = jnp.reshape(inputs, (bs, chs, -1))
        target = jnp.reshape(target, (bs, chs, -1))
        inputs = jnp.transpose(inputs, (0, 2, 1))
        target = jnp.transpose(target, (0, 2, 1))

    # Calculate loss components
    inputs_mag, inputs_phs, target_mag, target_phs = (
        jnp.ravel(inputs_mag),
        jnp.ravel(inputs_phs),
        jnp.ravel(target_mag),
        jnp.ravel(target_phs),
    )
    sc_mag_loss = (
        spectral_convergence_loss(inputs_mag, target_mag) * w_sc if w_sc else 0.0
    )
    log_mag_loss = (
        stft_magnitude_loss(
            inputs_mag, target_mag, log=True, reduction=reduction, distance=mag_distance
        )
        * w_log_mag
        if w_log_mag
        else 0.0
    )
    lin_mag_loss = (
        stft_magnitude_loss(
            inputs_mag,
            target_mag,
            log=False,
            reduction=reduction,
            distance=mag_distance,
        )
        * w_lin_mag
        if w_lin_mag
        else 0.0
    )
    phs_loss = (
        optax.squared_error(inputs_phs, target_phs).mean() * w_phs if w_phs else 0.0
    )

    # Combine loss components
    total_loss = sc_mag_loss + log_mag_loss + lin_mag_loss + phs_loss

    # Apply reduction (mean, sum)
    if reduction == "mean":
        total_loss = jnp.mean(total_loss)
    elif reduction == "sum":
        total_loss = jnp.sum(total_loss)

    # Return based on the output type
    if output == "loss":
        return total_loss
    elif output == "full":
        return total_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


def multi_resolution_stft_loss(
    traced_params,
    untraced_params,
    inputs,
    target,
    w_sc=1.0,
    w_log_mag=1.0,
    w_lin_mag=0.0,
    w_phs=0.0,
    scale=None,
    perceptual_weighting=None,
    eps=1e-8,
    output="loss",
    reduction="mean",
    mag_distance="L1",
):
    mrstft_loss = 0.0
    sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
    loss_fn = partial(
        stft_loss,
        inputs=inputs,
        target=target,
        w_sc=w_sc,
        w_log_mag=w_log_mag,
        w_lin_mag=w_lin_mag,
        w_phs=w_phs,
        scale=scale,
        perceptual_weighting=perceptual_weighting,
        eps=eps,
        output=output,
        reduction=reduction,
        mag_distance=mag_distance,
    )

    for p, q in zip(traced_params, untraced_params):
        if output == "full":
            tmp_loss = loss_fn(traced_params=p, untraced_params=q)
            mrstft_loss += tmp_loss[0]
            sc_mag_loss.append(tmp_loss[1])
            log_mag_loss.append(tmp_loss[2])
            lin_mag_loss.append(tmp_loss[3])
            phs_loss.append(tmp_loss[4])
        else:
            mrstft_loss += loss_fn(traced_params=p, untraced_params=q)

    mrstft_loss /= len(traced_params)

    if output == "loss":
        return mrstft_loss
    else:
        return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
