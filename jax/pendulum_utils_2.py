from typing import NamedTuple
import rl_types as RT
import jax_trajectory_utils_2 as JTU

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as jnp

import noise_procs as noise


dampedSpringNoise = noise.dampedSpringNoise


class PendulumParams(NamedTuple):
    g: float
    m: float
    l: float
    dt: float
    max_torque: float
    max_speed: float


default_pendulum_params = PendulumParams(10, 1, 1, 0.05, 2, 8)


def random_initial_state(
    key: noise.PRNGKey, params: PendulumParams = default_pendulum_params
):
    a, b = random.uniform(key, (2,))
    th = jnp.pi * (2 * a - 1)
    thdot = 8 * (2 * b - 1)
    return jnp.array([th, thdot])


@jit
def expand_state_cos_sin(S):
    th, thdot = S
    S_new = jnp.array([jnp.cos(th), jnp.sin(th), thdot])
    return S_new


@jit
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


@jit
def pendulum_step(state, u, params):
    th, thdot = state

    u = jnp.clip(u, -params.max_torque, params.max_torque)

    thdot = jnp.clip(
        thdot
        + (
            -3.0 * params.g / (2 * params.l) * jnp.sin(th + jnp.pi)
            + 3.0 / (params.m * params.l ** 2) * u
        )
        * params.dt,
        -params.max_speed,
        params.max_speed,
    )
    th = angle_normalize(th + thdot * params.dt)

    cost = th ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return jnp.array((th, thdot)).flatten(), -cost


def make_dyn_fn(params=default_pendulum_params) -> JTU.DynamicsFn:
    @jit
    def dyn_fn(S: RT.State, A: RT.Action) -> RT.SrTup:
        return pendulum_step(S, A, params)

    return dyn_fn
