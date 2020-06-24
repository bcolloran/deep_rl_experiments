from typing import Callable, NewType, Any, Tuple, TypeVar
from functools import partial
import jax.numpy as jnp
import jax
from jax import grad, jit, value_and_grad
import damped_spring_noise as dsn
from numpy.random import randn
from numpy import ndarray

dampedSpringNoise = dsn.dampedSpringNoise

# State = NewType("State", ndarray)
# Action = NewType("Action", ndarray)
# Noise = NewType("Noise", ndarray)
# NoiseState = NewType("NoiseState", ndarray)

# State: TypeAlias = ndarray
# Action: TypeAlias = ndarray
# Reward: TypeAlias = ndarray
# Noise: TypeAlias = ndarray
# NoiseState: TypeAlias = ndarray

# State = ndarray
# Action = ndarray
# Reward = ndarray
# Noise = ndarray
# NoiseState = ndarray

State = TypeVar("State", ndarray, jnp.ndarray)
Action = TypeVar("Action", ndarray, jnp.ndarray)
Reward = TypeVar("Reward", ndarray, jnp.ndarray)
Noise = TypeVar("Noise", ndarray, jnp.ndarray)
NoiseState = TypeVar("NoiseState", ndarray, jnp.ndarray)

DynamicsFn = Callable[[State, Action], Tuple[State, Reward]]
PolicyFn = Callable[[State], Action]
NoisePolicyFn = Callable[[State, Noise], Action]
NoiseFn = Callable[[NoiseState], Tuple[NoiseState, Noise]]

SarTup = Tuple[State, Action, Reward]
StateNoiseTup = Tuple[State, NoiseState]

StateScanTup = Tuple[State, SarTup]
StateScannableFn = Callable[[State, Any], StateScanTup]

SarTrajTup = Tuple[State, Action, Reward]


def make_scan_policy_dynamics_step_fn(
    dynamics_fn: DynamicsFn, policy_fn: PolicyFn
) -> StateScannableFn:
    @jit
    def episode_step(state: State, _) -> StateNoiseScanTup:
        action = policy_fn(state)
        state_next, reward = dynamics_fn(state, action)
        return state_next, (state, action, reward)

    return episode_step


@partial(jit, static_argnums=(0, 3))
def make_agent_episode(T, s0, episode_step_fn):
    _, traj = jax.lax.scan(episode_step_fn, s0, None, length=T)
    return traj


StateNoiseScanTup = Tuple[StateNoiseTup, SarTup]
StateNoiseScannableFn = Callable[[StateNoiseTup, Any], StateNoiseScanTup]


def make_scan_policy_dynamics_noise_step_fn(
    dynamics_fn: DynamicsFn, policy_fn: NoisePolicyFn, noise_fn: NoiseFn
) -> StateNoiseScannableFn:
    @jit
    def episode_step(carry: StateNoiseTup, _) -> StateNoiseScanTup:
        # print("carry", carry)
        state, noise_state = carry
        noise_state, eps = noise_fn(noise_state)
        action = policy_fn(state, eps)
        state_next, reward = dynamics_fn(state, action)
        # print("(state_next, noise_state)", (state_next, noise_state))
        return (state_next, noise_state), (state, action, reward)

    return episode_step


@partial(jit, static_argnums=(0, 3))
def make_agent_episode_noisy(
    T: int, S0: State, noise0: NoiseState, episode_step_fn: StateNoiseScannableFn
) -> SarTrajTup:
    _, traj = jax.lax.scan(episode_step_fn, (S0, noise0), None, length=T)
    return traj


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


def make_random_episode(
    T: int,
    dynamics_fn: DynamicsFn,
    action_noise_fn: NoiseFn,
    S0: State,
    noise0: NoiseState,
) -> SarTrajTup:
    random_policy = lambda state, noise: noise

    scan_fn = make_scan_policy_dynamics_noise_step_fn(
        dynamics_fn, random_policy, action_noise_fn
    )

    return make_agent_episode_noisy(T, S0, noise0, scan_fn)


def make_random_episode_fn(
    T: int,
    dynamics_fn: DynamicsFn,
    action_noise_fn: NoiseFn,
    S0_fn: Callable[[], State],
    noise0_fn: Callable[[], NoiseState],
) -> Callable[[], SarTrajTup]:
    random_policy = lambda state, noise: noise

    scan_fn = make_scan_policy_dynamics_noise_step_fn(
        dynamics_fn, random_policy, action_noise_fn
    )

    return lambda: make_agent_episode_noisy(T, S0_fn, noise0_fn, scan_fn)


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

