from typing import Callable, NewType, Any, Tuple, TypeVar, Union, Dict, Hashable
from functools import partial
import jax.numpy as jnp
import jax
from jax import grad, jit, value_and_grad
import noise_procs as noise
from numpy.random import randn
from numpy import ndarray
import rl_types as RT


# def make_scan_policy_dynamics_step_fn(
#     dynamics_fn: RT.DynamicsFn, policy_fn: RT.PolicyFn
# ) -> RT.StateScannableFn:
#     @jit
#     def episode_step(state: RT.State, _) -> RT.StateNoiseScanTup:
#         action = policy_fn(state)
#         state_next, reward = dynamics_fn(state, action)
#         return state_next, (state, action, reward)

#     return episode_step


# @partial(jit, static_argnums=(0, 3))
# def make_agent_episode(T, s0, episode_step_fn):
#     _, traj = jax.lax.scan(episode_step_fn, s0, None, length=T)
#     return traj


# def make_scan_policy_dynamics_noise_step_fn(
#     dynamics_fn: RT.DynamicsFn, policy_fn: RT.NoisePolicyFn, noise_fn: RT.NoiseFn
# ) -> RT.StateNoiseScannableFn:
#     @jit
#     def episode_step(carry: RT.StateNoiseTup, _) -> RT.StateNoiseScanTup:
#         # print("carry", carry)
#         state, noise_state = carry
#         noise_state, eps = noise_fn(noise_state)
#         action = policy_fn(state, eps)
#         state_next, reward = dynamics_fn(state, action)
#         # print("(state_next, noise_state)", (state_next, noise_state))
#         return (state_next, noise_state), (state, action, reward)

#     return episode_step


# @partial(jit, static_argnums=(0, 3))
# def make_agent_episode_noisy(
#     T: int,
#     S0: RT.State,
#     noise0: RT.NoiseState,
#     episode_step_fn: RT.StateNoiseScannableFn,
# ) -> RT.SarTrajTup:
#     _, traj = jax.lax.scan(episode_step_fn, (S0, noise0), None, length=T)
#     return traj


# def make_scan_dynamics_fn(dynamics_fn:DynamicsFn) ->St:
#     @jit
#     def episode_step(state, action):
#         state_next, reward = dynamics_fn(state, action)
#         return state_next, (state, action, reward)

#     return episode_step


# @partial(jit, static_argnums=(2,))
# def make_episode_with_actions_vect(S0, A, scan_dyn_fn):
#     _, traj = jax.lax.scan(scan_dyn_fn, S0, A)
#     return traj


# def make_random_episode(
#     T: int,
#     dynamics_fn: RT.DynamicsFn,
#     action_noise_fn: RT.NoiseFn,
#     S0: RT.State,
#     noise0: RT.NoiseState,
# ) -> RT.SarTrajTup:
#     random_policy = lambda state, noise: noise

#     scan_fn = make_scan_policy_dynamics_noise_step_fn(
#         dynamics_fn, random_policy, action_noise_fn
#     )

#     return make_agent_episode_noisy(T, S0, noise0, scan_fn)

DynamicsFn = Callable[[State, Action], Tuple[State, Reward]]
PolicyFn = Callable[[State], Action]
# NoisePolicyFn = Callable[[State, Noise], Action]
# NoisePolicyNetFn = Callable[[NNParams, State, Noise], Action]
PolicyNetFn = Callable[[PolicyNet, State, noise.Noise], Action]
StateInitFn = Callable[[noise.PRNGKey], RT.State]

StateNoiseCarryTup = Tuple[RT.State, noise.NoiseState]


def make_random_episode_fn(
    T: int,
    dynamics_fn: RT.DynamicsFn,
    random_action_fn: noise.NoiseFn,
    S0_fn: StateInitFn,
    noise0_fn: noise.NoiseInitFn,
) -> Callable[[], RT.SarTrajTup]:
    @jit
    def episode_step(
        carry: StateNoiseCarryTup, _: Any
    ) -> Tuple[StateNoiseCarryTup, RT.SarTup]:
        state, noise_state = carry
        noise_state, random_action = random_action_fn(noise_state)
        state_next, reward = dynamics_fn(state, random_action)
        return (state_next, noise_state), (state, random_action, reward)

    @jit
    def make_episode(key: noise.PRNGKey) -> RT.SarTrajTup:
        carry: StateNoiseCarryTup = (S0_fn(key), noise0_fn(key))
        _, traj = jax.lax.scan(episode_step, carry, None, length=T)
        return traj

    return make_episode


NNParamsStateNoiseCarryTup = Tuple[RT.NNParams, RT.State, noise.NoiseState]


def make_agent_policynet_episode_fn(
    T: int,
    policy_net_fn: RT.PolicyNetFn,
    dynamics_fn: RT.DynamicsFn,
    noise_fn: noise.NoiseFn,
    S0_fn: StateInitFn,
    noise0_fn: noise.NoiseInitFn,
) -> Callable[[RT.NNParams], RT.SarTrajTup]:
    @jit
    def episode_step(
        carry: NNParamsStateNoiseCarryTup, _: Any
    ) -> Tuple[NNParamsStateNoiseCarryTup, RT.SarTup]:
        nn_params, state, noise_state = carry
        noise_state, eps = noise_fn(noise_state)
        action = policy_net_fn(nn_params, state, eps)
        state_next, reward = dynamics_fn(state, action)
        return (nn_params, state_next, noise_state), (state, action, reward)

    @jit
    def make_episode(policy_net: RT.NNParams, key: noise.PRNGKey) -> RT.SarTrajTup:
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

