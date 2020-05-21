import numpy

import jax
from jax import grad, jit, vmap
import jax.numpy as np
from jax import random
from jax.ops import index, index_add, index_update

import matplotlib.pyplot as plt


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
    X = target_dim * numpy.random.rand(target_dim, batch_size) - 1
    return [(X[:, i : i + 1], target_fn(X[:, i : i + 1])) for i in range(batch_size)]


### MODEL FITTER


def plot_1d_model_fit(predict, params, target, Xs=None):
    if Xs is None:
        Xs = numpy.linspace(-1, 1, 1000)
    # plt.plot(Xs, predict(params, x).item() for x in Xs])
    plt.plot(Xs, predict(params, Xs.reshape(1, -1)).flatten())
    plt.plot(Xs, target(Xs))
    plt.show()


def fit_model(
    LR_0=0.1,
    LR_min=0.001,
    decay=0.99,
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

    if plot_fn is not None:
        plot_fn(predict, params, target_fn)

    for i in range(epoch_size * num_epochs):
        data = make_data_set(target_fn, target_fn_dim, batch_size)

        l = loss(params, data).item()
        loss_list.append(l)
        if np.isnan(l):
            print("loss is nan")
            break

        LR = max(LR_0 * decay ** i, LR_min)

        params = update(params, data, LR)
        print(f"epoch {i}, loss {l}, LR {LR}", end="\r")

        if i % epoch_size == 0:
            param_list.append(params)
            if plot_fn is not None:
                plot_fn(predict, params, target_fn)

    return param_list, loss_list
