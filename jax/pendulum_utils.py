from collections import namedtuple
from typing import NamedTuple
import numpy

# from numpy.random import rand

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as jnp

# from jax.ops import index, index_add, index_update

# from noise_procs import dampedSpringNoise
import noise_procs as noise

import jax_nn_utils as jnn
from jax_nn_utils import randKey
import rl_types as RT
import jax_trajectory_utils_2 as JTU

dampedSpringNoise = noise.dampedSpringNoise


# PendulumParams = namedtuple("Params", ["g", "m", "l", "dt", "max_torque", "max_speed"])


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


# @jit
# def expand_state_cos_sin(S):
#     th = S[:, 0]
#     thdot = S[:, 1]
#     return jnp.hstack([jnp.cos(th), jnp.sin(th), thdot,])


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


@jit
def controlledPendulumStep(state_and_params, u):
    th, thdot, params = state_and_params

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
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return jnp.array((th, thdot, cost))


@jit
def controlledPendulumStep_derivs(th, thdot, params=default_pendulum_params):
    d_thdot = (-3 * params.g / (2 * params.l) * jnp.sin(th + jnp.pi)) * params.dt
    newthdot = thdot + d_thdot
    d_th = newthdot * params.dt

    return jnp.array((d_th, d_thdot))


@jit
def controlledPendulumStep_scan(state_and_params, u):
    th, thdot, params = state_and_params

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
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return (th, thdot, params), jnp.array((th, thdot, cost))


@jit
def pendulumTraj_scan(th0, thdot0, uVect, params=default_pendulum_params):
    state = (th0, thdot0, params)
    _, traj = jax.lax.scan(controlledPendulumStep_scan, state, uVect)
    return traj


def make_random_episode(episode_len, params=default_pendulum_params, key=randKey()):
    theta0 = jnp.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 1)
    A = dampedSpringNoise(episode_len, key=randKey())

    traj = pendulumTraj_scan(theta0, thetadot0, A, params)

    S_th = angle_normalize(traj[:, 0:1])
    S_thdot = traj[:, 1:2]
    S = jnp.hstack([jnp.cos(S_th), jnp.sin(S_th), S_thdot])

    R = traj[:, 2:3]

    return (S, A.reshape(-1, 1), R)


U_opts = jnp.linspace(
    -default_pendulum_params.max_torque, default_pendulum_params.max_torque, 11
)
state_action_template = jnp.vstack(
    [jnp.ones_like(U_opts), jnp.ones_like(U_opts), jnp.ones_like(U_opts), U_opts]
)


###### DDQN


@jit
def ddqnRandomAction(cond_args):
    Q1, cos_th, sin_th, thdot, randkey = cond_args
    index = random.randint(randkey, (1,), 0, len(U_opts))[0]
    return U_opts[index]


@jit
def ddqnBestAction(Q1, cos_th, sin_th, thdot):
    # (val_est, randkey) = val_est_randkey
    s_a = state_action_template * jnp.array([[cos_th, sin_th, thdot, 1]]).T
    val_ests = jnn.predict(Q1, s_a)
    # NOTE: we use argMIN here since everything is framed as cost not reward
    # Note also that this does not need to be clipped; all the opts are in
    # the right range
    return U_opts[jnp.argmin(val_ests)]


ddqnBestActionsVect = jax.vmap(ddqnBestAction, (None, 0, 0, 0), (0))


@jit
def ddqnBestAction_wrap(cond_args):
    Q1, cos_th, sin_th, thdot, key = cond_args
    return ddqnBestAction(Q1, cos_th, sin_th, thdot)


@jit
def controlledPendulumStepDDQN(state_and_params, __ignore):
    th, thdot, params, Q1, eps, key = state_and_params

    key, subkey = random.split(key)

    u = jax.lax.cond(
        random.uniform(subkey) > eps,
        ddqnBestAction_wrap,
        ddqnRandomAction,
        (Q1, jnp.cos(th), jnp.sin(th), thdot, key),
    )
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
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return (th, thdot, params, Q1, eps, key), jnp.array((u, cost, th, thdot))


# @jit
def pendulumTrajDDQN(th0, thdot0, T, Q1, eps, params=default_pendulum_params):
    key = randKey()
    state = (th0, thdot0, params, Q1, eps, key)
    _, u_cost_sNext = jax.lax.scan(controlledPendulumStepDDQN, state, None, length=T)
    S = jnp.vstack([jnp.array([th0, thdot0]), u_cost_sNext[:-1, 2:4]])
    A = u_cost_sNext[:, 0]
    Cost = u_cost_sNext[:, 1]
    return (jnp.transpose(S), jnp.transpose(A), jnp.transpose(Cost))


###### END DDQN


def cost_of_state(th, thdot, params, u=None):
    if u is None:
        u = numpy.zeros_like(th)
    return angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)


@jit
def costOfTrajectory(traj):
    return jnp.sum(traj[:, 2])


@jit
def pendulumTrajCost_scan(th0, thdot0, uVect, params=default_pendulum_params):
    traj = pendulumTraj_scan(th0, thdot0, uVect, params)
    return costOfTrajectory(traj)


trajectoryGrad_scan = jax.jit(grad(pendulumTrajCost_scan, argnums=2))
trajectoryValAndGrad_scan = jax.jit(value_and_grad(pendulumTrajCost_scan, argnums=2))


def make_n_step_traj_episode(
    episode_len, n, stdizer, params,
):
    theta0 = jnp.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    traj = jnp.array(
        pendulumTraj_scan(
            theta0,
            thetadot0,
            jnp.clip(
                dampedSpringNoise(episode_len + n, key=randKey()),
                -params.max_torque,
                params.max_torque,
            ),
            params,
        )
    )
    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]

    stdizer.observe_reward_vec(traj[:, 2])

    R = jnp.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]
    episode = jnp.hstack([S_th, S_thdot, S_n_th, S_n_thdot, R,])
    return episode


def make_n_step_sin_cos_traj_episode(episode_len, n, stdizer, params, key=randKey()):
    theta0 = jnp.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    traj = jnp.array(
        pendulumTraj_scan(
            theta0,
            thetadot0,
            jnp.clip(
                dampedSpringNoise(episode_len + n, key=randKey()),
                -params.max_torque,
                params.max_torque,
            ),
            params,
        )
    )
    stdizer.observe_reward_vec(traj[:, 2])

    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]
    R = jnp.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]

    episode = jnp.hstack(
        [
            jnp.cos(S_th),
            jnp.sin(S_th),
            S_thdot,
            jnp.cos(S_n_th),
            jnp.sin(S_n_th),
            S_n_thdot,
            R,
        ]
    )
    return episode


def make_n_step_sarsa_episode(episode_len, n, stdizer, params, key=randKey()):
    theta0 = jnp.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    u = jnp.clip(
        dampedSpringNoise(episode_len + n, key=randKey()),
        -params.max_torque,
        params.max_torque,
    )
    traj = jnp.array(pendulumTraj_scan(theta0, thetadot0, u, params,))
    stdizer.observe_reward_vec(traj[:, 2])

    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]
    R = jnp.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]

    episode = jnp.hstack(
        [
            jnp.cos(S_th),
            jnp.sin(S_th),
            S_thdot,
            u[:-n].reshape(-1, 1),
            jnp.cos(S_n_th),
            jnp.sin(S_n_th),
            S_n_thdot,
            u[n:].reshape(-1, 1),
            R,
        ]
    )
    return episode


def make_n_step_data_from_DDQN(S, A, C, n, stdizer, params, key=randKey()):
    S_0_th = angle_normalize(S[0, :-n])
    S_0_thdot = S[1, 0:-n]

    S_n_th = angle_normalize(S[0, n:])
    S_n_thdot = S[1, n:]

    A_0 = A[:-n]
    A_n = A[n:]

    stdizer.observe_reward_vec(C)
    R = jnp.vstack([stdizer.standardize_reward(C[m : -(n - m)]) for m in range(n)])

    episode = jnp.vstack(
        [
            jnp.cos(S_0_th),
            jnp.sin(S_0_th),
            S_0_thdot,
            A_0,
            jnp.cos(S_n_th),
            jnp.sin(S_n_th),
            S_n_thdot,
            A_n,
            R,
        ]
    )
    return episode


# ######################
# OPTIMIZATION
# ######################


def get_best_random_control(best_so_far, th, thdot, params, random_starts, horizon):
    best_traj_cost, best_traj, best_u = best_so_far
    for j in range(random_starts):
        # uVect = 2*numpy.random.randn(T-t)
        key = random.PRNGKey(int(numpy.random.rand(1) * 1000000))
        uVect = jnp.clip(
            dampedSpringNoise(horizon, sigma=0.2, theta=0.005, phi=0.2, key=key),
            -params.max_torque,
            params.max_torque,
        )
        traj = pendulumTraj_scan(th, thdot, uVect, params)
        traj_cost = costOfTrajectory(traj)
        if traj_cost < best_traj_cost:
            best_traj_cost = traj_cost
            best_traj = traj
            best_u = uVect
    return best_traj_cost, best_traj, best_u


def optimize_control(best_so_far, th, thdot, params, opt_iters, LR):
    best_traj_cost, best_traj, best_u = best_so_far
    uVect = best_u
    for i in range(opt_iters):
        dgdu = trajectoryGrad_scan(th, thdot, uVect, params)
        uVect = uVect - LR * (0.99 ** i) * dgdu / jnp.linalg.norm(dgdu)
        traj = pendulumTraj_scan(th, thdot, uVect, params)
        traj_cost = costOfTrajectory(traj)
        if traj_cost < best_traj_cost:
            best_traj_cost = traj_cost
            best_traj = traj
            best_u = uVect
    return best_traj_cost, best_traj, best_u


# ######################
# PLOTTING
# ######################
