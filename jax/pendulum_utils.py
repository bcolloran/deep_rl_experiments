from collections import namedtuple
import numpy
from numpy.random import rand

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as np

# from jax.ops import index, index_add, index_update

import damped_spring_noise

from jax_nn_utils import randKey

PendulumParams = namedtuple("Params", ["g", "m", "l", "dt", "max_torque", "max_speed"])

default_pendulum_params = PendulumParams(10, 1, 1, 0.05, 2, 8)

dampedSpringNoiseJit = damped_spring_noise.dampedSpringNoiseJit


@jit
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


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


@jit
def controlledPendulumStep_derivs(th, thdot, params=default_pendulum_params):
    d_thdot = (-3 * params.g / (2 * params.l) * np.sin(th + np.pi)) * params.dt
    newthdot = thdot + d_thdot
    d_th = newthdot * params.dt

    return np.array((d_th, d_thdot))


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

    # th = angle_normalize(traj[:, 0:1])
    # thdot = traj[:, 1]
    S_th = angle_normalize(traj[0:-n, 0:1])
    S_thdot = traj[0:-n, 1:2]

    stdizer.observe_reward_vec(traj[:, 2])

    # S_th = traj[0:-n, 0:2]
    R = np.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    # S_n = traj[n:, 0:2]
    S_n_th = angle_normalize(traj[n:, 0:1])
    S_n_thdot = traj[n:, 1:2]
    # episode = np.hstack([S, S_n, R])
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
    # traj[:, 0] = angle_normalize(traj[:, 0])
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
