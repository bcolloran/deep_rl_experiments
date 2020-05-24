import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class Plotter1d(object):
    def __init__(self, target_fn, Xs=None, jupyter=True):
        self.target_fn = target_fn

        if Xs is None:
            self.Xs = np.linspace(-1, 1, 100)

        fig, (axFit, axLoss) = plt.subplots(1, 2, figsize=(20, 5))

        self.fig = fig
        self.axFit = axFit
        self.axLoss = axLoss
        self.jupyter = True if jupyter else False

    def update_plot(self, params, loss_list, batch, Xs=None):
        if loss_list == []:
            loss_list = [0]
        self.fig.suptitle(f"batch {batch}, loss {loss_list[-1]}", fontsize=14)

        self.axFit.clear()
        self.axFit.plot(self.Xs, predict(params, self.Xs.reshape(1, -1)).flatten())
        self.axFit.plot(self.Xs, self.target_fn(self.Xs))

        self.axLoss.clear()
        self.axLoss.plot(loss_list)
        self.axLoss.set_yscale("log")
        self.axLoss.set_yscale("log")

        if self.jupyter:
            clear_output(wait=True)
            display(self.fig)


class Plotter2d(object):
    def __init__(self, target_fn, n_grid=100, jupyter=True):
        self.n_grid = n_grid

        X1, X2 = np.meshgrid(np.linspace(-1, 1, n_grid), np.linspace(-1, 1, n_grid))

        X = np.vstack([X1.ravel(), X2.ravel()])
        self.plotX = X
        self.Y = target_fn(X).reshape(self.n_grid, self.n_grid)

        fig, (axLoss, axY, axYhat, axResid) = plt.subplots(1, 4, figsize=(16, 4))

        self.fig = fig
        self.axY = axY
        self.imY = self.axY.imshow(self.Y, origin="lower", cmap="PiYG", alpha=0.9)
        self.axY.set_title("target")
        self.cbarY = self.fig.colorbar(self.imY, ax=self.axY)

        self.axYhat = axYhat
        self.imYhat = self.axYhat.imshow(
            np.zeros((n_grid, n_grid)), origin="lower", cmap="PiYG", alpha=0.9
        )
        self.axYhat.set_title("prediction")
        self.cbarYhat = self.fig.colorbar(self.imYhat, ax=self.axYhat)

        self.axResid = axResid
        self.imResid = self.axResid.imshow(
            np.zeros((n_grid, n_grid)), origin="lower", cmap="PiYG", alpha=0.9
        )
        self.axResid.set_title("residual")
        self.cbarResid = self.fig.colorbar(self.imResid, ax=self.axResid)

        # self.axResid = axResid
        self.axLoss = axLoss
        self.jupyter = True if jupyter else False

    def update_plot(self, Yhat, loss_list, batch_num):

        Yhat = Yhat.reshape(self.n_grid, self.n_grid)

        if loss_list == []:
            loss_list = [0]
        self.fig.suptitle(f"batch {batch_num}, loss {loss_list[-1]:.4f}", fontsize=14)

        self.imYhat.set_data(Yhat)
        self.imYhat.set_clim(vmin=Yhat.min(), vmax=Yhat.max())

        Resid = self.Y - Yhat
        self.imResid.set_data(Resid)
        self.imResid.set_clim(vmin=Resid.min(), vmax=Resid.max())

        self.axLoss.clear()
        self.axLoss.plot(loss_list)
        self.axLoss.set_yscale("log")

        plt.draw()

        if self.jupyter:
            # plt.show()
            clear_output(wait=True)
            display(self.fig)
