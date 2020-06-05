from collections import namedtuple
import numpy

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as np

# from jax.ops import index, index_add, index_update

import damped_spring_noise

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

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
def pendulumTraj_scan(th0, thdot0, uVect, params):
    state = (th0, thdot0, params)
    _, traj = jax.lax.scan(controlledPendulumStep_scan, state, uVect)
    return traj


@jit
def costOfTrajectory(traj):
    return np.sum(traj[:, 2])


@jit
def pendulumTrajCost_scan(th0, thdot0, uVect, params):
    traj = pendulumTraj_scan(th0, thdot0, uVect, params)
    return costOfTrajectory(traj)


trajectoryGrad_scan = jax.jit(grad(pendulumTrajCost_scan, argnums=2))
trajectoryValAndGrad_scan = jax.jit(value_and_grad(pendulumTrajCost_scan, argnums=2))


@jit
def controlledPendulumStep_derivs(th, thdot, params):
    d_thdot = (-3 * params.g / (2 * params.l) * np.sin(th + np.pi)) * params.dt
    newthdot = thdot + d_thdot
    d_th = newthdot * params.dt

    return np.array((d_th, d_thdot))


def plotTrajInPhaseSpace(traj, params, uVect=None):
    X = np.linspace(-2 * np.pi, 2 * np.pi, 21)
    Y = np.arange(-params.max_speed - 2, params.max_speed + 2, 1)

    U, V = np.meshgrid(X, Y)
    gridshape = U.shape

    THDOT, THDOTDOT = controlledPendulumStep_derivs(U.ravel(), V.ravel(), params)

    THDOT = THDOT.reshape(gridshape)
    THDOTDOT = THDOTDOT.reshape(gridshape)

    fig, (axPhase, axU) = plt.subplots(1, 2, figsize=(12, 5))
    axPhase.quiver(X, Y, THDOT, THDOTDOT)
    axPhase.plot(traj[:, 0], traj[:, 1])
    axPhase.plot(traj[0, 0], traj[0, 1], "o")

    if uVect is not None:
        axU.plot(uVect)

    plt.show()


def pendulum_traj_animation(traj, uVect, th0):
    fig, ax = plt.subplots()

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_aspect(1)

    (line,) = ax.plot([], [], lw=2)
    (line2,) = ax.plot([], [], lw=2)

    trajectory = traj[:, 0]
    inputs = uVect

    def init():
        line.set_data([0, np.sin(th0)], [0, np.cos(th0)])
        line2.set_data([0, 0], [-1.5, -1.5])
        return (line, line2)

    def animate(i):
        tip_x = np.sin(trajectory[i])
        tip_y = np.cos(trajectory[i])

        line.set_data([0, tip_x], [0, tip_y])
        line2.set_data(
            [tip_x, tip_x + np.sin(trajectory[i] + np.pi / 2) * inputs[i]],
            [tip_y, tip_y + np.cos(trajectory[i] + np.pi / 2) * inputs[i]],
        )
        # line2.set_data([0, inputs[i]], [-1.5,-1.5])
        return (line, line2)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(trajectory),
        interval=1000 * 0.05,
        blit=True,
    )

    return HTML(anim.to_jshtml())


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
