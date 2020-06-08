from collections import namedtuple
import numpy
from numpy.random import rand

import jax
from jax import grad, value_and_grad, random, jit
import jax.numpy as np

# from jax.ops import index, index_add, index_update

import damped_spring_noise

from jax_nn_utils import randKey

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display, clear_output

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


@jit
def controlledPendulumStep_derivs(th, thdot, params=default_pendulum_params):
    d_thdot = (-3 * params.g / (2 * params.l) * np.sin(th + np.pi)) * params.dt
    newthdot = thdot + d_thdot
    d_th = newthdot * params.dt

    return np.array((d_th, d_thdot))


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


class PendulumValuePlotter(object):
    def __init__(
        self,
        predict,
        model,
        n_grid=100,
        jupyter=True,
        pend_params=default_pendulum_params,
        show_init_est=True,
    ):
        self.n_grid = n_grid

        X1, X2 = np.meshgrid(
            np.pi * np.linspace(-1, 1, n_grid), 8 * np.linspace(-1, 1, n_grid)
        )

        X = np.vstack([X1.ravel(), X2.ravel()])
        self.plotX = X

        # fig, (axLoss, axY, axYhat, axResid) = plt.subplots(1, 4, figsize=(8, 8))
        # fig, (axY, axYhat) = plt.subplots(1, 2, figsize=(14, 7))
        fig, (axLoss, axY, axYhat) = plt.subplots(1, 3, figsize=(15, 5))
        self.fig = fig
        self.axY = axY
        if show_init_est:
            self.Y = predict(model, X).reshape(self.n_grid, self.n_grid)
            self.imY = self.axY.imshow(
                self.Y,
                origin="lower",
                cmap="PiYG",
                alpha=0.9,
                extent=(-np.pi, np.pi, -8, 8),
                aspect=0.5,
            )
            self.cbarY = self.fig.colorbar(self.imY, ax=self.axY)
        self.axY.set_title("initial NN prediction")

        self.axYhat = axYhat
        self.imYhat = self.axYhat.imshow(
            np.zeros((n_grid, n_grid)),
            origin="lower",
            cmap="PiYG",
            alpha=0.9,
            extent=(-np.pi, np.pi, -8, 8),
            aspect=0.5,
        )
        self.axYhat.set_title("prediction")
        self.cbarYhat = self.fig.colorbar(self.imYhat, ax=self.axYhat)

        U, V = np.meshgrid(np.pi * np.linspace(-1, 1, 17), 8 * np.linspace(-1, 1, 17))
        THDOT, THDOTDOT = controlledPendulumStep_derivs(
            U.ravel(), V.ravel(), pend_params
        )
        THDOT = THDOT.reshape(U.shape)
        THDOTDOT = THDOTDOT.reshape(U.shape)
        axYhat.quiver(U, V, THDOT, THDOTDOT)
        # axYhat.set_aspect(1)

        # self.axResid = axResid
        # self.imResid = self.axResid.imshow(
        #     np.zeros((n_grid, n_grid)), origin="lower", cmap="PiYG", alpha=0.9
        # )
        # self.axResid.set_title("residual")
        # self.cbarResid = self.fig.colorbar(self.imResid, ax=self.axResid)

        # self.axResid = axResid
        self.axLoss = axLoss
        self.jupyter = True if jupyter else False

    def update_plot(self, Yhat, loss_list, epoch_num, title=None):

        Yhat = Yhat.reshape(self.n_grid, self.n_grid)

        if loss_list == []:
            loss_list = [0]
        if title is None:
            title = f"epoch {epoch_num}, loss {loss_list[-1]:.4f}"
        self.fig.suptitle(title, fontsize=14)

        self.imYhat.set_data(Yhat)
        self.imYhat.set_clim(vmin=Yhat.min(), vmax=Yhat.max())

        # Resid = self.Y - Yhat
        # self.imResid.set_data(Resid)
        # self.imResid.set_clim(vmin=Resid.min(), vmax=Resid.max())

        self.axLoss.clear()
        self.axLoss.plot(loss_list)
        self.axLoss.set_yscale("log")

        plt.draw()

        if self.jupyter:
            # plt.show()
            clear_output(wait=True)
            display(self.fig)


class PendulumValuePlotter2(object):
    def __init__(
        self, n_grid=100, jupyter=True, pend_params=default_pendulum_params,
    ):
        self.axes_ready = False
        self.n_grid = n_grid
        self.pend_params = pend_params
        grid_pts = np.linspace(-1, 1, n_grid)

        X1, X2 = np.meshgrid(np.pi * grid_pts, 8 * grid_pts)
        X = np.vstack([X1.ravel(), X2.ravel()])
        self.plotX1 = X1.ravel()
        self.plotX2 = X2.ravel()
        self.plotX = X

        self.jupyter = True if jupyter else False

    def reshape_imdata(self, data):
        return data.reshape(self.n_grid, self.n_grid)

    def init_axes(self, im_plots, line_plots, title):
        self.axes_ready = True

        num_plots = (len(line_plots) if line_plots is not None else 0) + (
            len(im_plots) if im_plots is not None else 0
        )

        num_rows = int(np.ceil(num_plots / 4))
        num_cols = int(np.floor(num_plots / num_rows))

        fig, ax_table = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows)
        )
        ax_list = [ax for row in ax_table for ax in row]
        self.fig = fig
        # self.axY = axY
        self.axes = {}
        self.images = {}
        ax_index = 0

        U, V = np.meshgrid(np.pi * np.linspace(-1, 1, 17), 8 * np.linspace(-1, 1, 17))
        THDOT, THDOTDOT = controlledPendulumStep_derivs(
            U.ravel(), V.ravel(), self.pend_params
        )
        THDOT = THDOT.reshape(U.shape)
        THDOTDOT = THDOTDOT.reshape(U.shape)

        if line_plots is not None:
            for plot_name, plot_data in line_plots:
                self.axes[plot_name] = ax_list[ax_index]
                self.axes[plot_name].set_title(plot_name)
                ax_index += 1

        if im_plots is not None:
            for plot_name, plot_data in im_plots:
                self.axes[plot_name] = ax_list[ax_index]
                self.axes[plot_name].set_title(plot_name)
                ax_index += 1

                self.images[plot_name] = self.axes[plot_name].imshow(
                    self.reshape_imdata(plot_data),
                    origin="lower",
                    cmap="PiYG",
                    alpha=0.9,
                    extent=(-np.pi, np.pi, -8, 8),
                    aspect=0.5,
                )
                self.axes[plot_name].set_title(plot_name)
                self.fig.colorbar(self.images[plot_name], ax=self.axes[plot_name])
                self.axes[plot_name].quiver(U, V, THDOT, THDOTDOT, pivot="mid")

        if title is not None:
            self.fig.suptitle(title, fontsize=14)

    def update_plot(self, im_plots=None, line_plots=None, title=None):
        if not self.axes_ready:
            self.init_axes(im_plots, line_plots, title)

        if title is not None:
            self.fig.suptitle(title, fontsize=14)

        if im_plots is not None:
            for plot_name, plot_data in im_plots:
                self.images[plot_name].set_data(self.reshape_imdata(plot_data))
                self.images[plot_name].set_clim(
                    vmin=plot_data.min(), vmax=plot_data.max()
                )
        if line_plots is not None:
            for plot_name, plot_data in line_plots:
                self.axes[plot_name].clear()
                for line_name, line_data in plot_data:
                    self.axes[plot_name].plot(line_data)
                self.axes[plot_name].set_yscale("log")
                self.axes[plot_name].set_title(plot_name)

        plt.draw()

        if self.jupyter:
            # plt.show()
            clear_output(wait=True)
            display(self.fig)


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
