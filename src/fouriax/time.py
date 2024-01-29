import jax.numpy as jnp


def log_cosh_loss(inputs, target, a=1.0, eps=1e-8):
    losses = jnp.mean((1 / a) * jnp.log(jnp.cosh(a * (inputs - target)) + eps), axis=-2)
    losses = jnp.mean(losses)
    return losses


def esr_loss(inputs, target, eps=1e-8):
    num = jnp.sum(((target - inputs) ** 2), axis=1)
    denom = jnp.sum(target**2, axis=1) + eps
    losses = num / denom
    losses = jnp.mean(losses)
    return losses


def dc_loss(inputs, target, eps=1e-8):
    num = jnp.mean(target - inputs, axis=1) ** 2
    denom = jnp.mean(target**2, axis=1) + eps
    losses = num / denom
    losses = jnp.mean(losses)
    return losses
