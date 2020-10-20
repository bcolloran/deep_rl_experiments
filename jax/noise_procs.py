from jax import random, jit, lax
import jax.numpy as jnp
from functools import partial
import rl_types as RT
from typing import (
    Callable,
    NewType,
    Any,
    Tuple,
    TypeVar,
    Union,
    Dict,
    Hashable,
    NamedTuple,
)
import jax


Noise = RT.Tensor
PRNGKey = NewType("PRNGKey", RT.Tensor)


class NoiseState(NamedTuple):
    key: PRNGKey
    state: Any


NoiseStepOut = Tuple[NoiseState, Noise]
NoiseInitFn = Callable[[PRNGKey], NoiseState]
NoiseFn = Callable[[NoiseState], NoiseStepOut]


@jit
def next_key(key: PRNGKey) -> PRNGKey:
    return random.split(key, 1)[0]


def normalNoiseInit(key: PRNGKey) -> NoiseState:
    return NoiseState(key, None)


@jit
def normalStep(x: NoiseState) -> NoiseStepOut:
    eps: Noise = random.normal(x.key)
    return NoiseState(next_key(x.key), None), eps


def mvNormalNoiseInit(key: PRNGKey,) -> NoiseState:
    return NoiseState(key=key, state=None)


@partial(jit, static_argnums=(1,))
def mvNormalStep(x: NoiseState, dim=1) -> NoiseStepOut:
    eps: Noise = random.normal(x.key, shape=(dim,))
    return NoiseState(key=next_key(x.key), state=None), eps


def dampedSpringNoiseStateInit(
    key, dim=1, x=None, v=None, sigma=0.5, theta=0.05, phi=0.01
) -> NoiseState:
    if x is None:
        x = jnp.zeros(dim)
    if v is None:
        v = jnp.zeros(dim)
    return NoiseState(key=next_key(key), state=(x, v, sigma, theta, phi))


@jit
def dampedSpringNoiseStep(stepIn: NoiseState, _=None) -> NoiseStepOut:
    x, v, sigma, theta, phi = stepIn.state
    v += -theta * x - phi * v + sigma * random.normal(stepIn.key, shape=x.shape)
    x += v
    return (
        NoiseState(key=next_key(stepIn.key), state=(x, v, sigma, theta, phi)),
        x,
    )


@partial(jit, static_argnums=(0,))
def dampedSpringNoise(T, noise_params=(0.5, 0.05, 0.01), key=random.PRNGKey(0)):
    sigma, theta, phi = noise_params
    state = (0.0, 0.0, sigma, theta, phi, key)
    _, noise = lax.scan(dampedSpringNoiseStep, state, None, length=T)
    return noise

