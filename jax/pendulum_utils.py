from collections import namedtuple
import numpy
from numpy.random import rand

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as np

# from jax.ops import index, index_add, index_update

import damped_spring_noise

import jax_nn_utils as jnn
from jax_nn_utils import randKey

PendulumParams = namedtuple("Params", ["g", "m", "l", "dt", "max_torque", "max_speed"])

default_pendulum_params = PendulumParams(10, 1, 1, 0.05, 2, 8)

dampedSpringNoiseJit = damped_spring_noise.dampedSpringNoiseJit


@jit
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@jit
def controlledPendulumStep(state_and_params, u):
    th, thdot, params = state_and_params

    u = np.clip(u, -params.max_torque, params.max_torque)

    thdot = np.clip(
        thdot
        + (
            -3.0 * params.g / (2 * params.l) * np.sin(th + np.pi)
            + 3.0 / (params.m * params.l ** 2) * u
        )
        * params.dt,
        -params.max_speed,
        params.max_speed,
    )
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return np.array((th, thdot, cost))


@jit
def controlledPendulumStep_derivs(th, thdot, params=default_pendulum_params):
    d_thdot = (-3 * params.g / (2 * params.l) * np.sin(th + np.pi)) * params.dt
    newthdot = thdot + d_thdot
    d_th = newthdot * params.dt

    return np.array((d_th, d_thdot))


@jit
def controlledPendulumStep_scan(state_and_params, u):
    th, thdot, params = state_and_params

    u = np.clip(u, -params.max_torque, params.max_torque)

    thdot = np.clip(
        thdot
        + (
            -3.0 * params.g / (2 * params.l) * np.sin(th + np.pi)
            + 3.0 / (params.m * params.l ** 2) * u
        )
        * params.dt,
        -params.max_speed,
        params.max_speed,
    )
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    # newthdot = newthdot, -max_speed, max_speed)

    return (th, thdot, params), np.array((th, thdot, cost))


@jit
def pendulumTraj_scan(th0, thdot0, uVect, params=default_pendulum_params):
    state = (th0, thdot0, params)
    _, traj = jax.lax.scan(controlledPendulumStep_scan, state, uVect)
    return traj


U_opts = np.linspace(
    -default_pendulum_params.max_torque, default_pendulum_params.max_torque, 11
)
state_action_template = np.vstack(
    [np.ones_like(U_opts), np.ones_like(U_opts), np.ones_like(U_opts), U_opts]
)


# @jit
# def ddqnBestAction(val_est_randkey):
#     (val_est, randkey) = val_est_randkey
#     # NOTE: we use argMIN here since everything is framed as cost not reward
#     # Note also that this does not need to be clipped; all the opts are in
#     # the right range
#     return U_opts[np.argmin(val_est)]


@jit
def ddqnRandomAction(cond_args):
    Q1, cos_th, sin_th, thdot, randkey = cond_args
    index = random.randint(randkey, (1,), 0, len(U_opts))[0]
    return U_opts[index]


@jit
def ddqnBestAction(Q1, cos_th, sin_th, thdot):
    # (val_est, randkey) = val_est_randkey
    s_a = state_action_template * np.array([[cos_th, sin_th, thdot, 1]]).T
    val_ests = jnn.predict(Q1, s_a)
    # NOTE: we use argMIN here since everything is framed as cost not reward
    # Note also that this does not need to be clipped; all the opts are in
    # the right range
    return U_opts[np.argmin(val_ests)]


ddqnBestActionsVect = jax.vmap(ddqnBestAction, (None, 0, 0, 0), (0))


@jit
def ddqnBestAction_wrap(cond_args):
    Q1, cos_th, sin_th, thdot, key = cond_args
    return ddqnBestAction(Q1, cos_th, sin_th, thdot)


@jit
def controlledPendulumStepDDQN(state_and_params, __ignore):
    th, thdot, params, Q1, eps, key = state_and_params
    # s_a = state_action_template * np.array([[np.cos(th), np.sin(th), thdot, 1]]).T

    # val_ests = jnn.predict(Q1, s_a)

    key, subkey = random.split(key)

    u = jax.lax.cond(
        random.uniform(subkey) > eps,
        ddqnBestAction_wrap,
        ddqnRandomAction,
        (Q1, np.cos(th), np.sin(th), thdot, key),
    )
    thdot = np.clip(
        thdot
        + (
            -3.0 * params.g / (2 * params.l) * np.sin(th + np.pi)
            + 3.0 / (params.m * params.l ** 2) * u
        )
        * params.dt,
        -params.max_speed,
        params.max_speed,
    )
    th = th + thdot * params.dt

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return (th, thdot, params, Q1, eps, key), np.array((u, cost, th, thdot))


# @jit
def pendulumTrajDDQN(th0, thdot0, T, Q1, eps, params=default_pendulum_params):
    key = randKey()
    state = (th0, thdot0, params, Q1, eps, key)
    _, u_cost_sNext = jax.lax.scan(controlledPendulumStepDDQN, state, None, length=T)
    S = np.vstack([np.array([th0, thdot0]), u_cost_sNext[:-1, 2:4]])
    A = u_cost_sNext[:, 0]
    Cost = u_cost_sNext[:, 1]
    return (np.transpose(S), np.transpose(A), np.transpose(Cost))


def cost_of_state(th, thdot, params, u=None):
    if u is None:
        u = numpy.zeros_like(th)
    return angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)


@jit
def costOfTrajectory(traj):
    return np.sum(traj[:, 2])


@jit
def pendulumTrajCost_scan(th0, thdot0, uVect, params=default_pendulum_params):
    traj = pendulumTraj_scan(th0, thdot0, uVect, params)
    return costOfTrajectory(traj)


trajectoryGrad_scan = jax.jit(grad(pendulumTrajCost_scan, argnums=2))
trajectoryValAndGrad_scan = jax.jit(value_and_grad(pendulumTrajCost_scan, argnums=2))


def make_n_step_traj_episode(
    episode_len, n, stdizer, params,
):
    theta0 = np.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    traj = np.array(
        pendulumTraj_scan(
            theta0,
            thetadot0,
            np.clip(
                dampedSpringNoiseJit(episode_len + n, key=randKey()),
                -params.max_torque,
                params.max_torque,
            ),
            params,
        )
    )
    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]

    stdizer.observe_reward_vec(traj[:, 2])

    R = np.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]
    episode = np.hstack([S_th, S_thdot, S_n_th, S_n_thdot, R,])
    return episode


def make_n_step_sin_cos_traj_episode(episode_len, n, stdizer, params, key=randKey()):
    theta0 = np.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    traj = np.array(
        pendulumTraj_scan(
            theta0,
            thetadot0,
            np.clip(
                dampedSpringNoiseJit(episode_len + n, key=randKey()),
                -params.max_torque,
                params.max_torque,
            ),
            params,
        )
    )
    stdizer.observe_reward_vec(traj[:, 2])

    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]
    R = np.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]

    episode = np.hstack(
        [
            np.cos(S_th),
            np.sin(S_th),
            S_thdot,
            np.cos(S_n_th),
            np.sin(S_n_th),
            S_n_thdot,
            R,
        ]
    )
    return episode


def make_n_step_sarsa_episode(episode_len, n, stdizer, params, key=randKey()):
    theta0 = np.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    u = np.clip(
        dampedSpringNoiseJit(episode_len + n, key=randKey()),
        -params.max_torque,
        params.max_torque,
    )
    traj = np.array(pendulumTraj_scan(theta0, thetadot0, u, params,))
    stdizer.observe_reward_vec(traj[:, 2])

    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]
    R = np.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]

    episode = np.hstack(
        [
            np.cos(S_th),
            np.sin(S_th),
            S_thdot,
            u[:-n].reshape(-1, 1),
            np.cos(S_n_th),
            np.sin(S_n_th),
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
    R = np.vstack([stdizer.standardize_reward(C[m : -(n - m)]) for m in range(n)])

    episode = np.vstack(
        [
            np.cos(S_0_th),
            np.sin(S_0_th),
            S_0_thdot,
            A_0,
            np.cos(S_n_th),
            np.sin(S_n_th),
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
        uVect = np.clip(
            dampedSpringNoiseJit(horizon, sigma=0.2, theta=0.005, phi=0.2, key=key),
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
        uVect = uVect - LR * (0.99 ** i) * dgdu / np.linalg.norm(dgdu)
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
