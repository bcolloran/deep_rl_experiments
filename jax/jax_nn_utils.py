import numpy

import jax
from jax import grad, jit, vmap
import jax.numpy as np
from jax import random
from jax.ops import index, index_add, index_update

import matplotlib.pyplot as plt
from IPython.display import display, clear_output


@jit
def relu(x):
    return np.maximum(0, x)


@jit
def normalize(x):
    return x / np.linalg.norm(x)


def alternate(N, a, b):
    i = 0
    out = b
    while i < N:
        i += 1
        out = a if out == b else b
        yield out


def alternating_array(N, a, b):
    return numpy.array(list(alternate(N, a, b)))


def checkerboard(N, M):
    return numpy.outer(alternating_array(N, -1.0, 1.0), alternating_array(M, 1.0, -1.0))


###  INITIALIZATION


def init_network_params_randn(sizes, scale=1e-2):
    return [
        (scale * numpy.random.randn(n, m), scale * numpy.random.randn(n, 1))
        for m, n in zip(sizes[:-1], sizes[1:])
    ]


def init_network_params_unif(sizes, scale=1):
    return [
        (
            scale * 2 * numpy.random.rand(n, m) - 1,
            scale * 2 * numpy.random.rand(n, 1) - 1,
        )
        for m, n in zip(sizes[:-1], sizes[1:])
    ]


def init_network_params_He(sizes):
    return [
        (
            numpy.random.randn(n, m) * numpy.sqrt(2 / m),
            numpy.zeros((n, 1))
            # numpy.random.randn(n, 1)*numpy.sqrt(2/m)
        )
        for m, n in zip(sizes[:-1], sizes[1:])
    ]


def init_network_params_He_offset(sizes):
    return [
        (
            numpy.random.randn(n, m) * numpy.sqrt(2 / m),
            # numpy.linspace(-.3,-1,n).reshape(-1,1)
            numpy.random.randn(n, 1)
            # numpy.random.randn(n, 1)*numpy.sqrt(2/m)
        )
        for m, n in zip(sizes[:-1], sizes[1:])
    ]


def init_network_params_zeros(sizes):
    return [
        (numpy.zeros((n, m)), numpy.zeros((n, 1)))
        for m, n in zip(sizes[:-1], sizes[1:])
    ]


### PREDICTION AND OPTIMIZATION


@jit
def predict(params, x):
    # per-example predictions
    for w, b in params[:-1]:
        # x = w @ x
        # x = x + b
        # x = relu(x)
        x = relu(np.dot(w, x) + b)
    w, b = params[-1]
    return np.dot(w, x) + b


@jit
def loss(params, data):
    return np.sum([(y - predict(params, x)) ** 2 for x, y in data])


@jit
def update(params, data, LR):
    grads = grad(loss)(params, data)
    return [(w - LR * dw, b - LR * db) for (w, b), (dw, db) in zip(params, grads)]


@jit
def update_normalize_grad(params, data, LR):
    grads = grad(loss)(params, data)
    return [
        (w - LR * normalize(dw), b - LR * normalize(db))
        for (w, b), (dw, db) in zip(params, grads)
    ]


def make_data_set(target_fn, target_dim, batch_size):
    X = 2 * numpy.random.rand(target_dim, batch_size) - 1
    return [(X[:, i : i + 1], target_fn(X[:, i : i + 1])) for i in range(batch_size)]


### MODEL FITTER


class Plotter1d(object):
    def __init__(self, target_fn, Xs=None, jupyter=True):
        # self.predict = predict
        self.target_fn = target_fn

        if Xs is None:
            self.Xs = numpy.linspace(-1, 1, 100)

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
            # plt.show()

            display(self.fig)
            # clear_output(wait=True)
            # plt.show()


def fit_model(
    LR=0.001,
    LR_min=0.001,
    decay=1,
    epoch_size=100,
    num_epochs=10,
    batch_size=64,
    target_fn=lambda x: x,
    target_fn_dim=1,
    layers=[None, 80, 40, 20, 1],
    layer_initializer=init_network_params_He,
    params=None,
    plot_fn=None,
):
    LR_0 = LR
    layers[0] = target_fn_dim

    loss_list = []
    param_list = []
    if params is None:
        params = layer_initializer(layers)

    if plot_fn is not None:
        epoch = 0
        plot_fn(params, loss_list, epoch)

    for i in range(epoch_size * num_epochs):
        data = make_data_set(target_fn, target_fn_dim, batch_size)

        l = loss(params, data).item()
        loss_list.append(l)
        if np.isnan(l):
            print("loss is nan")
            break

        LR = max(LR_0 * decay ** i, LR_min)

        params = update(params, data, LR)
        print(f"batch {i}, loss {l}, LR {LR}", end="\r")

        if (i + 1) % epoch_size == 0:
            # print("epoch")
            param_list.append(params)
            if plot_fn is not None:
                # print("should plot")
                plot_fn(params, loss_list, i)

    return param_list, loss_list


# @jit
def update_adam(params, data, ms, vs, t, LR, beta1=0.9, beta2=0.99, eps=1e-8):
    # https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
    grads = grad(loss)(params, data)

    ms = [
        (beta1 * m_w + (1 - beta1) * dw, beta1 * m_b + (1 - beta1) * db)
        for (m_w, m_b), (dw, db) in zip(ms, grads)
    ]

    vs = [
        (beta2 * v_w + (1 - beta2) * dw ** 2, beta2 * v_b + (1 - beta2) * db ** 2)
        for (v_w, v_b), (dw, db) in zip(vs, grads)
    ]

    t = t + 1
    # param_updates = [
    #     (
    #         LR * (m_w / (1 - beta1 ** t) / (v_w / (np.sqrt(1 - beta2 ** t) + eps))),
    #         LR * (m_b / (1 - beta1 ** t) / (v_b / (np.sqrt(1 - beta2 ** t) + eps))),
    #     )
    #     for (m_w, m_b), (v_w, v_b) in zip(ms, vs)
    # ]
    # print("param_updates", param_updates)
    # print("param_updates", (param_updates[0][0][0, 0].item()))
    mHats = [(m_w / (1 - beta1 ** t), m_b / (1 - beta1 ** t)) for (m_w, m_b) in ms]

    vHats = [(v_w / (1 - beta2 ** t), v_b / (1 - beta2 ** t)) for (v_w, v_b) in vs]

    param_updates = [
        (LR * mHat_w / (np.sqrt(vHat_w) + eps), LR * mHat_b / (np.sqrt(vHat_b) + eps),)
        for (mHat_w, mHat_b), (vHat_w, vHat_b) in zip(mHats, vHats)
    ]

    # for (w_pu, b_pu) in param_updates:
    #     if np.sum(w_pu ** 2) == 0:
    #         print("np.sum(w_pu**2)==0")

    #     if np.sum(b_pu ** 2) == 0:
    #         print("np.sum(b_pu**2)==0")

    #     if numpy.isnan(np.sum(w_pu ** 2)):
    #         print("numpy.isnan(ip.sum(w_pu**2)")

    #     if numpy.isnan(np.sum(b_pu ** 2)):
    #         print("numpy.isnan(np.sum(b_pu**2)")
    # vHatss = [
    #     (beta2 * v_w + (1 - beta2) * dw ** 2, beta2 * v_b + (1 - beta2) * db ** 2)
    #     for (v_w, v_b), (dw, db) in zip(vs, grads)
    # ]
    return (
        [(w - w_pu, b - b_pu) for (w, b), (w_pu, b_pu) in zip(params, param_updates)],
        ms,
        vs,
    )


def fit_model_adam(
    LR=0.0001,
    # LR_min=0.001,
    # decay=0.99,
    epoch_size=100,
    num_epochs=10,
    batch_size=64,
    target_fn=lambda x: x,
    target_fn_dim=1,
    layers=[None, 80, 40, 20, 1],
    layer_initializer=init_network_params_He,
    params=None,
    plot_fn=None,
):
    layers[0] = target_fn_dim

    loss_list = []
    param_list = []
    if params is None:
        params = layer_initializer(layers)

    ms = init_network_params_zeros(layers)
    vs = init_network_params_zeros(layers)

    if plot_fn is not None:
        epoch = 0
        plot_fn(params, loss_list, epoch)

    for i in range(epoch_size * num_epochs):
        data = make_data_set(target_fn, target_fn_dim, batch_size)

        l = loss(params, data).item()
        loss_list.append(l)
        if np.isnan(l):
            print("loss is nan")
            break

        # LR = max(LR_0 * decay ** i, LR_min)

        params, ms, vs = update_adam(params, data, ms, vs, i, LR)
        # print(f"epoch {i}, loss {l}, LR {LR}", end="\r")

        if (i + 1) % epoch_size == 0:
            param_list.append(params)
            if plot_fn is not None:
                plot_fn(params, loss_list, i)

    return param_list, loss_list
