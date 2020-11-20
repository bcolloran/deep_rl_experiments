import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display, clear_output

from pendulum_utils import controlledPendulumStep_derivs, default_pendulum_params


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
        self,
        n_grid=100,
        jupyter=True,
        pend_params=default_pendulum_params,
        panel_size=5,
    ):
        self.panel_size = panel_size
        self.axes_ready = False
        self.n_grid = n_grid
        self.pend_params = pend_params
        grid_pts = np.linspace(-1, 1, n_grid)

        X1, X2 = np.meshgrid(np.pi * grid_pts, 8 * grid_pts)

        self.plotX1 = X1.ravel()
        self.plotX2 = X2.ravel()
        self.plotX = np.vstack([self.plotX1, self.plotX2])

        self.jupyter = True if jupyter else False

    def reshape_imdata(self, data):
        return data.reshape(self.n_grid, self.n_grid)

    def init_axes(self, im_plots, line_plots, traj_plots, title):
        self.axes_ready = True

        num_plots = (
            (len(line_plots) if line_plots is not None else 0)
            + (len(im_plots) if im_plots is not None else 0)
            + (len(traj_plots) if traj_plots is not None else 0)
        )

        MAX_NUM_COLUMNS = 4
        num_rows = int(np.ceil(num_plots / MAX_NUM_COLUMNS))
        num_cols = int(np.ceil(num_plots / num_rows))

        fig, ax_table = plt.subplots(
            num_rows,
            num_cols,
            figsize=(self.panel_size * num_cols, self.panel_size * num_rows),
        )
        try:

            if len(ax_table.shape) == 2:
                ax_list = [ax for row in ax_table for ax in row]
            if len(ax_table.shape) == 1:
                ax_list = [ax for ax in ax_table]
        except AttributeError:
            ax_list = [ax_table]
        self.fig = fig
        # self.axY = axY
        self.axes = {}
        self.axes_lines = {}
        self.images = {}
        ax_index = 0

        U, V = np.meshgrid(np.pi * np.linspace(-1, 1, 17), 8 * np.linspace(-1, 1, 17))
        THDOT, THDOTDOT = controlledPendulumStep_derivs(
            U.ravel(), V.ravel(), self.pend_params
        )
        THDOT = THDOT.reshape(U.shape)
        THDOTDOT = THDOTDOT.reshape(U.shape)

        if line_plots is not None:
            for plot_info in line_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                plot_kwargs = None
                if len(plot_info) == 3:
                    plot_kwargs = plot_info[2]
                ax = ax_list[ax_index]
                ax_index += 1
                self.axes[plot_name] = ax
                self.axes_lines[plot_name] = {}
                ax.set_title(plot_name)
                for line_name, line_data in plot_data:
                    self.axes_lines[plot_name][line_name] = ax.plot(
                        line_data, label=line_name
                    )[0]
                box = ax.get_position()
                if len(plot_data) < 5:
                    # legend on bottom
                    ax.set_position(
                        [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
                    )
                    ax.legend(
                        loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5,
                    )
                else:
                    # Put a legend to the right of the current axis
                    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                if plot_kwargs is not None and "yscale" in plot_kwargs:
                    ax.set_yscale(plot_kwargs["yscale"])
                else:
                    ax.set_yscale("symlog")
                ax.set_title(plot_name)

        if im_plots is not None:
            for plot_info in im_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                plot_kwargs = None
                if len(plot_info) == 3:
                    plot_kwargs = plot_info[2]
                self.axes[plot_name] = ax_list[ax_index]
                ax_index += 1
                self.axes[plot_name].set_title(plot_name)

                if plot_kwargs is not None and plot_kwargs["diverging"]:
                    cmap = "PRGn"
                else:
                    cmap = "magma"

                self.images[plot_name] = self.axes[plot_name].imshow(
                    self.reshape_imdata(plot_data),
                    origin="lower",
                    cmap=cmap,
                    alpha=0.9,
                    extent=(-np.pi, np.pi, -8, 8),
                    aspect=0.5,
                )
                self.fig.colorbar(self.images[plot_name], ax=self.axes[plot_name])
                self.axes[plot_name].quiver(U, V, THDOT, THDOTDOT, pivot="mid")

        if traj_plots is not None:
            for plot_info in traj_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                plot_kwargs = None
                if len(plot_info) == 3:
                    plot_kwargs = plot_info[2]
                ax = ax_list[ax_index]
                self.axes[plot_name] = ax
                ax_index += 1
                ax.set_title(plot_name)
                ax.quiver(U, V, THDOT, THDOTDOT, pivot="mid")
                self.axes_lines[plot_name] = {}
                for line_name, line_x, line_y in plot_data:
                    self.axes_lines[plot_name][line_name] = ax.plot(
                        line_x, line_y, marker=".", label=line_name
                    )[0]

        if title is not None:
            self.fig.suptitle(title, fontsize=14)

    def update_plot(self, im_plots=None, line_plots=None, traj_plots=None, title=None):
        if not self.axes_ready:
            self.init_axes(im_plots, line_plots, traj_plots, title)

        if title is not None:
            self.fig.suptitle(title, fontsize=14)

        if im_plots is not None:
            for plot_info in im_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                self.images[plot_name].set_data(self.reshape_imdata(plot_data))
                self.images[plot_name].set_clim(
                    vmin=plot_data.min(), vmax=plot_data.max()
                )
        if line_plots is not None:
            for plot_info in line_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                for line_name, line_data in plot_data:
                    self.axes_lines[plot_name][line_name].set_data(
                        range(len(line_data)), line_data
                    )
                self.axes[plot_name].relim()
                self.axes[plot_name].autoscale_view()

        if traj_plots is not None:
            for plot_info in traj_plots:
                plot_name, plot_data = plot_info[0], plot_info[1]
                for line_name, line_x, line_y in plot_data:
                    line_x, line_y = process_X_wrapping_series(line_x, line_y)
                    self.axes_lines[plot_name][line_name].set_data(line_x, line_y)
                # self.axes[plot_name].autoscale_view()

        plt.draw()

        if self.jupyter:
            clear_output(wait=True)
            display(self.fig)

        return self.fig


def process_X_wrapping_series(xs, ys, eps=0.5):
    xs2 = []
    ys2 = []
    for i, x in enumerate(xs):
        if i == 0:
            xs2.append(x)
            ys2.append(ys[i])
            continue
        if abs(x - xs[i - 1]) > eps:
            xs2.append(np.nan)
            ys2.append(np.nan)
        xs2.append(x)
        ys2.append(ys[i])
    return xs2, ys2


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
