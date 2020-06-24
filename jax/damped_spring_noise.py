from jax import random, jit, lax
import jax.numpy as jnp
from functools import partial


@jit
def normalStep(key):
    key, subkey = random.split(key)
    return key, random.normal(subkey)


def dampedSpringNoiseStateInit(
    dim=1, seed=0, x=None, v=None, sigma=0.5, theta=0.05, phi=0.01
):
    key = random.PRNGKey(seed)
    if x is None:
        x = jnp.zeros(dim)
    if v is None:
        v = jnp.zeros(dim)
    return (x, v, sigma, theta, phi, key)


@jit
def dampedSpringNoiseStep(state, __ARG_NOT_USED=None):
    x, v, sigma, theta, phi, key = state
    key, subkey = random.split(key)
    v += -theta * x - phi * v + sigma * random.normal(subkey)
    x += v
    return (x, v, sigma, theta, phi, key), x


@partial(jit, static_argnums=(0,))
def dampedSpringNoise(T, noise_params=(0.5, 0.05, 0.01), key=random.PRNGKey(0)):
    sigma, theta, phi = noise_params
    state = (0.0, 0.0, sigma, theta, phi, key)
    _, noise = lax.scan(dampedSpringNoiseStep, state, None, length=T)
    return noise
