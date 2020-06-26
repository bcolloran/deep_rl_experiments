from typing import Callable, NewType, Any, Tuple, TypeVar, Union, Dict, Hashable
from functools import partial
import jax.numpy as jnp
import jax
from jax import grad, jit, value_and_grad
import noise_procs as noise
from numpy.random import randn
from numpy import ndarray

# import rl_types as
from rl_types import State, Action, Reward, SarTup, SarTrajTup, NNParams

DynamicsFn = Callable[[State, Action], Tuple[State, Reward]]
PolicyFn = Callable[[State], Action]
PolicyNetFn = Callable[[NNParams, State, noise.Noise], Action]

StateInitFn = Callable[[noise.PRNGKey], State]

StateNoiseCarryTup = Tuple[State, noise.NoiseState]


def make_random_episode_fn(
    T: int,
    dynamics_fn: DynamicsFn,
    random_action_fn: noise.NoiseFn,
    S0_fn: StateInitFn,
    noise0_fn: noise.NoiseInitFn,
) -> Callable[[noise.PRNGKey], SarTrajTup]:
    @jit
    def episode_step(
        carry: StateNoiseCarryTup, _: Any
    ) -> Tuple[StateNoiseCarryTup, SarTup]:
        state, noise_state = carry
        noise_state, random_action = random_action_fn(noise_state)
        state_next, reward = dynamics_fn(state, random_action)
        return (state_next, noise_state), (state, random_action, reward)

    @jit
    def make_episode(key: noise.PRNGKey) -> SarTrajTup:
        carry: StateNoiseCarryTup = (S0_fn(key), noise0_fn(key))
        _, traj = jax.lax.scan(episode_step, carry, None, length=T)
        return traj

    return make_episode


NNParamsStateNoiseCarryTup = Tuple[NNParams, State, noise.NoiseState]


def make_agent_policynet_episode_fn(
    T: int,
    policy_net_fn: PolicyNetFn,
    dynamics_fn: DynamicsFn,
    noise_fn: noise.NoiseFn,
    S0_fn: StateInitFn,
    noise0_fn: noise.NoiseInitFn,
) -> Callable[[NNParams, noise.PRNGKey], SarTrajTup]:
    @jit
    def episode_step(
        carry: NNParamsStateNoiseCarryTup, _: Any
    ) -> Tuple[NNParamsStateNoiseCarryTup, SarTup]:
        nn_params, state, noise_state = carry
        noise_state, eps = noise_fn(noise_state)
        action = policy_net_fn(nn_params, state, eps)
        state_next, reward = dynamics_fn(state, action)
        return (nn_params, state_next, noise_state), (state, action, reward)

    @jit
    def make_episode(policy_net: NNParams, key: noise.PRNGKey) -> SarTrajTup:
        carry: NNParamsStateNoiseCarryTup = (policy_net, S0_fn(key), noise0_fn(key))
        _, traj = jax.lax.scan(episode_step, carry, None, length=T)
        return traj

    return make_episode


# def make_random_episode(
#     T,
#     scan_dyn_fn,
#     state_shape,
#     noise_fn=dampedSpringNoise,
#     noise_params=None,
#     key=jax.random.PRNGKey(0),
# ):
#     if noise_params is None:
#         A = noise_fn(T, key=key)
#     else:
#         A = noise_fn(T, noise_params, key=key)
#     S0 = randn(state_shape)
#     return make_episode_with_actions_vect(S0, A, scan_dyn_fn)

