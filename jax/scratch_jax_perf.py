# %%
import numpy as np
from numpy.random import randn, rand
from functools import partial

import time
import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Relu, Dense, FanOut
from jax.nn.initializers import he_normal, zeros, ones

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import OrderedDict as odict


from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
import jax_trajectory_utils as jtu
from noise_procs import dampedSpringNoise
import pendulum_utils as PU
import pendulum_plotting as PP
import simple_logger as SL
import SAC_agent as SAC
import replay_buffer as RB

reload(PU)
reload(PP)
reload(SL)
reload(SAC)
reload(RB)
reload(jnn)
reload(jtu)

get_ipython().run_line_magic("matplotlib", "inline")

params = PU.default_pendulum_params
#%%

dyn_scan_fn = jtu.make_scan_dynamics_fn(pendulum_step)


T = 1000
S0 = np.array((0.1, 0.1))

A = dampedSpringNoise(T)
S, A, R = jtu.make_episode_with_actions_vect(S0, A, params, dyn_scan_fn)


t0 = time.time()
for i in range(50):
    A = dampedSpringNoise(T)
    S, A, R = jtu.make_episode_with_actions_vect(S0, A, params, dyn_scan_fn)
print(f"elapsed: {time.time()-t0}")


def make_scan_dynamics_fn(dynamics_fn, params):
    @jit
    def episode_step(state, action):
        state_next, reward = dynamics_fn(state, action, params)
        return state_next, (state, action, reward)

    return episode_step


@partial(jit, static_argnums=(2,))
def make_episode_with_actions_vect(S0, A, scan_dyn_fn):
    # scan_dyn_fn = make_scan_dynamics_fn(dynamics_fn)
    # carry = (S0, params)
    _, traj = jax.lax.scan(scan_dyn_fn, S0, A)
    return traj


dyn_scan_fn = jit(make_scan_dynamics_fn(pendulum_step, params))

S, A, R = make_episode_with_actions_vect(S0, A, dyn_scan_fn)
t0 = time.time()
for i in range(50):
    A = dampedSpringNoise(T)
    S, A, R = make_episode_with_actions_vect(S0, A, dyn_scan_fn)
    # _, traj = jax.lax.scan(scan_dyn_fn, S0, A)
print(f"elapsed: {time.time()-t0}")

