import numpy

# import jax
from jax import grad, jit, value_and_grad
import jax.numpy as np

from jax import random

# from jax.ops import index, index_add, index_update


def randKey():
    return random.PRNGKey(int(100000 * numpy.random.rand(1)))


@jit
def relu(x):
    return np.maximum(0, x)


@jit
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_sample(x):
    return softmax_sample_jit(x, numpy.random.rand(1))


@jit
def softmax_sample_jit(x, eps):
    prob = softmax(x)
    cumprob = 0
    for i, p in enumerate(prob):
        cumprob += p
        if cumprob > eps:
            return i


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


#  INITIALIZATION


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


def init_network_params_ones(sizes):
    return [
        (numpy.ones((n, m)), numpy.ones((n, 1))) for m, n in zip(sizes[:-1], sizes[1:])
    ]


def copy_network(params):
    return [(w.copy(), b.copy()) for (w, b) in params]


@jit
def interpolate_networks(params_current, params_goal, tau):
    return [
        (tau * w + (1 - tau) * w2, tau * b + (1 - tau) * b2)
        for (w, b), (w2, b2) in zip(params_goal, params_current)
    ]


### PREDICTION AND OPTIMIZATION


@jit
def predict(params, x):
    # per-example predictions
    for w, b in params[:-1]:
        # x = relu(np.dot(w, x) + b)
        x = relu(w @ x + b)
    w, b = params[-1]
    return np.dot(w, x) + b


@jit
def loss(params, data):
    X, Y = data
    return np.mean((Y - predict(params, X)) ** 2)


@jit
def update(params, data, LR):
    grads = grad(loss)(params, data)
    return [(w - LR * dw, b - LR * db) for (w, b), (dw, db) in zip(params, grads)]


@jit
def update_value_TD0(V_params, R, S, S_next, discount, LR):
    grads = grad(predict)(V_params, S)
    TDerr = LR * (R + discount * predict(V_params, S_next) - predict(V_params, S))
    return [
        (w - TDerr * dw, b - TDerr * db) for (w, b), (dw, db) in zip(V_params, grads)
    ]


@jit
def update_and_loss(params, data, LR):
    value, grads = value_and_grad(loss)(params, data)
    return (
        [(w - LR * dw, b - LR * db) for (w, b), (dw, db) in zip(params, grads)],
        value,
    )


@jit
def update_normalize_grad(params, data, LR):
    grads = grad(loss)(params, data)
    return [
        (w - LR * normalize(dw), b - LR * normalize(db))
        for (w, b), (dw, db) in zip(params, grads)
    ]


def make_data_set(target_fn, target_dim, batch_size):
    X = 2 * numpy.random.rand(target_dim, batch_size) - 1
    Y = target_fn(X)
    return X, Y


class relu_MLP:
    def __init__(self, sizes):
        self.params = init_network_params_He(sizes)

    def predict(self, data):
        return predict(self.params, data)

    def batch_update(self, data_batch, LR=0.001):
        self.params = update(self.params, data_batch, LR)


### MODEL FITTER


def fit_model_to_target_fn(
    LR=0.001,
    LR_min=0.0,
    decay=1,
    epoch_size=100,
    num_epochs=10,
    batch_size=64,
    target_fn=lambda x: x,
    target_fn_dim=1,
    layers=[None, 80, 40, 20, 1],
    layer_initializer=init_network_params_He,
    params=None,
    plotter=None,
):
    layers[0] = target_fn_dim

    loss_list = []
    param_list = []
    if params is None:
        params = layer_initializer(layers)

    if plotter is not None:
        Yhat = predict(params, plotter.plotX)
        print(Yhat)
        plotter.update_plot(Yhat, loss_list, batch_num=0)

    for i in range(epoch_size * num_epochs):
        data = make_data_set(target_fn, target_fn_dim, batch_size)

        l = loss(params, data).item()
        loss_list.append(l)
        if np.isnan(l):
            print("loss is nan")
            break

        LR = max(LR * decay, LR_min)

        params = update(params, data, LR)
        print(f"batch {i}, loss {l}, LR {LR}", end="\r")

        if (i + 1) % epoch_size == 0:
            # print("epoch")
            param_list.append(params)
            if plotter is not None:
                # print("should plot")
                Yhat = predict(params, plotter.plotX)
                plotter.update_plot(Yhat, loss_list, i)

    return param_list, loss_list


@jit
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

    # NOTE what the heck is this"t=t+1"? dbl check source material
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


def fit_model_to_target_fn_adam(
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


class RewardStandardizer:
    # uses combined variance formula here:
    # https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html
    def __init__(
        self, min_normalization_std=0.0001,
    ):
        self.min_normalization_std = min_normalization_std

        self.observed_reward_mean = 0
        self.observed_reward_std = 0
        self.var_agg = 0
        self.num_rewards_observed = 0

    def load_reward_dict(self, data_dict):
        self.observed_reward_mean = data_dict["observed_reward_mean"]
        self.observed_reward_std = data_dict["observed_reward_std"]
        self.var_agg = data_dict["var_agg"]
        self.num_rewards_observed = data_dict["num_rewards_observed"]
        self.min_normalization_std = data_dict["min_normalization_std"]

    def get_reward_dict(self):
        data_dict = {}
        data_dict["observed_reward_mean"] = self.observed_reward_mean
        data_dict["observed_reward_std"] = self.observed_reward_std
        data_dict["var_agg"] = self.var_agg
        data_dict["num_rewards_observed"] = self.num_rewards_observed
        data_dict["min_normalization_std"] = self.min_normalization_std
        return data_dict

    def standardize_reward(self, reward):
        # Note: for rewards, we ONLY rescale, not recenter--
        # moving the mean can change some
        # rewards from positive to negative (or vice versa), which can
        # change an agent's "will to live" -- i.e., if a small negative rewards
        # at each step is put in place to encourage an agent to try to end episodes
        # quickly, recententering s.t. this ends up positive will encourage agents
        # to stay alive indefinitely (or vice versa)
        if self.num_rewards_observed == 0:
            raise ValueError(
                "num_rewards_observed==0; must observe at least one reward before standardizing"
            )
        return reward / self.observed_reward_std

    def observe_reward(self, reward):
        self.num_rewards_observed += 1

        next_mean = (
            self.observed_reward_mean
            + (reward - self.observed_reward_mean) / self.num_rewards_observed
        )

        self.var_agg += (reward - next_mean) * (reward - self.observed_reward_mean)

        self.observed_reward_mean = next_mean

        self.observed_reward_std = np.maximum(
            np.sqrt(self.var_agg / self.num_rewards_observed),
            self.min_normalization_std,
        )

    def observe_reward_vec(self, rewards):
        n1 = np.prod(rewards.shape)
        mu1 = np.mean(rewards)
        n1S1 = np.var(rewards) * n1

        n2 = self.num_rewards_observed
        mu2 = self.observed_reward_mean
        n2S2 = self.var_agg

        next_mean = (n1 * mu1 + n2 * mu2) / (n1 + n2)

        self.var_agg = (
            n1S1 + n2S2 + n1 * (mu1 - next_mean) ** 2 + n2 * (mu2 - next_mean) ** 2
        )

        self.num_rewards_observed = n1 + n2

        self.observed_reward_mean = next_mean

        self.observed_reward_std = np.maximum(
            np.sqrt(self.var_agg / self.num_rewards_observed),
            self.min_normalization_std,
        )
