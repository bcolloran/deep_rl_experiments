from jax import random, jit, lax
from functools import partial


@jit
def dampedSpringNoiseStep(state, __ARG_NOT_USED=None):
    x, v, sigma, theta, phi, key = state
    key, subkey = random.split(key)
    v += -theta * x - phi * v + sigma * random.normal(subkey)
    x += v
    return (x, v, sigma, theta, phi, key), x


@partial(jit, static_argnums=(0,))
def dampedSpringNoise(T, sigma=0.5, theta=0.05, phi=0.01, key=random.PRNGKey(0)):
    state = (0.0, 0.0, sigma, theta, phi, key)
    _, noise = lax.scan(dampedSpringNoiseStep, state, None, length=T)
    return noise
