import numpy

import jax
from jax import grad, jit, value_and_grad, tree_multimap, tree_map
import jax.numpy as jnp

from jax import random


def copy_network(params):
    return tree_map(lambda x: x.copy(), params)


@jit
def interpolate_networks(params_current, params_goal, tau):
    return tree_multimap(
        lambda x1, x2: tau * x1 + (1 - tau) * x2, params_goal, params_current
    )


@jit
def add_gradient(nn_params, g, LR):
    return jax.tree_multimap(lambda x, dx: x + LR * dx, nn_params, g)


@jit
def add_networks(n1, n2):
    return jax.tree_multimap(jnp.add, n1, n2)


@jit
def zeros_like_network(net):
    return jax.tree_map(jnp.zeros_like, net)


### PREDICTION AND OPTIMIZATION


@jit
def predict(params, x):
    # per-example predictions
    for w, b in params[:-1]:
        # x = relu(jnp.dot(w, x) + b)
        x = relu(w @ x + b)
    w, b = params[-1]
    return jnp.dot(w, x) + b


@jit
def loss(params, data):
    X, Y = data
    return jnp.mean((Y - predict(params, X)) ** 2)


@jit
def update(params, data, LR):
    grads = grad(loss)(params, data)
    return [(w - LR * dw, b - LR * db) for (w, b), (dw, db) in zip(params, grads)]


@jit
def update_and_loss(params, data, LR):
    value, grads = value_and_grad(loss)(params, data)
    return (
        [(w - LR * dw, b - LR * db) for (w, b), (dw, db) in zip(params, grads)],
        value,
    )
