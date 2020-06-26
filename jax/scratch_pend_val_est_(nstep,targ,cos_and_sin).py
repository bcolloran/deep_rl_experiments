# %%
import numpy as np
from numpy.random import randn, rand

# import time
# import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as jnp

# from jax.ops import index, index_add, index_update

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import OrderedDict as odict


from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
from noise_procs import dampedSpringNoise
import pendulum_utils as PU
import pendulum_plotting as PP
import simple_logger as SL

reload(PU)
reload(PP)
reload(SL)
reload(jnn)


get_ipython().run_line_magic("matplotlib", "inline")

params = PU.default_pendulum_params


# %%# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# NOTE: now trying TD(n)
# NOTE: now trying cos,sin state representation


episode_len = 300
num_epochs = 200
episodes_per_epoch = 10
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3

discount = 0.97

tau = 0.002  # best: ~0.005?

LR_0 = 0.00003  # best ~1e-5?
decay = 0.993

num_obs_rewards = 4

layers = [3, 80, 80, 1]
vn1 = jnn.init_network_params_He(layers)
vn1_targ = jnn.copy_network(vn1)
vn2 = jnn.init_network_params_He(layers)
vn2_targ = jnn.copy_network(vn2)

plotter = PP.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotX = np.vstack([np.cos(plotter.plotX1), np.sin(plotter.plotX1), plotter.plotX2])
stdizer = jnn.RewardStandardizer()

L = SL.Logger()

grad_log_names = []
for l_num in range(len(layers) - 1):
    for l_type in ["w", "b"]:
        for model_num in ["1", "2"]:
            grad_log_names.append(f"d{l_type}{l_num}_vn{model_num}")

for i in range(num_epochs):
    epoch_memory = None
    for j in range(episodes_per_epoch):
        episode = PU.make_n_step_sin_cos_traj_episode(
            episode_len, num_obs_rewards, stdizer, params
        )
        if epoch_memory is None:
            epoch_memory = episode
        else:
            epoch_memory = np.vstack([epoch_memory, episode])

    sample_order = np.random.permutation(epoch_memory.shape[0])
    LR = LR_0 * decay ** i
    for j in range(int(samples_per_epoch / batch_size)):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = jnp.transpose(epoch_memory[sample_indices, :])

        S = batch[0:3, :]

        S_lookahead = batch[3:6, :]

        Y = (discount ** num_obs_rewards) * np.minimum(
            jnn.predict(vn1_targ, S_lookahead), jnn.predict(vn2_targ, S_lookahead)
        )

        L.epoch_avg.obs("mean pred(vn*,S_n)*d^n", np.mean(Y))

        R = np.zeros_like(Y)
        for t in range(num_obs_rewards):
            R += (discount ** t) * batch[(6 + t) : (7 + t), :]
        Y += R

        L.epoch_avg.obs("mean Y", np.mean(Y))
        L.epoch_avg.obs("mean R", np.mean(R))
        L.epoch_avg.obs("mean predict(vn1,S)", np.mean(jnn.predict(vn1, S)))
        L.epoch_avg.obs("mean predict(vn2,S)", np.mean(jnn.predict(vn2, S)))

        vn1, loss1, norm_grads1 = jnn.update_with_loss_and_norm_grad(vn1, (S, Y), LR)
        L.epoch_avg.obs("loss1", loss1)
        for l_num, (dw, db) in enumerate(norm_grads1):
            L.epoch_avg.obs(f"dw{l_num}_vn1", dw)
            L.epoch_avg.obs(f"db{l_num}_vn1", db)

        vn2, loss2, norm_grads2 = jnn.update_with_loss_and_norm_grad(vn2, (S, Y), LR)
        L.epoch_avg.obs("loss2", loss2)
        for l_num, (dw, db) in enumerate(norm_grads2):
            L.epoch_avg.obs(f"dw{l_num}_vn2", dw)
            L.epoch_avg.obs(f"db{l_num}_vn2", db)

        if (j + 1) % update_every == 0:
            vn1_targ = jnn.interpolate_networks(vn1_targ, vn1, tau)
            vn2_targ = jnn.interpolate_networks(vn2_targ, vn2, tau)

    vn1_pred = jnn.predict(vn1, plotX)
    vn2_pred = jnn.predict(vn2, plotX)
    vn1_targ_pred = jnn.predict(vn1_targ, plotX)
    vn2_targ_pred = jnn.predict(vn2_targ, plotX)
    vn1_targ_err = vn1_pred - vn1_targ_pred
    vn2_targ_err = vn2_pred - vn2_targ_pred

    im_plots = [
        ("vn1 prediction", vn1_pred),
        ("vn2 prediction", vn2_pred),
        ("vn1_targ prediction", vn1_targ_pred),
        ("vn2_targ prediction", vn2_targ_pred),
        ("vn1 - vn1_targ", vn1_targ_err),
        ("vn2 - vn2_targ", vn2_targ_err),
    ]

    L.end_epoch()
    L.epoch.obs("MSE(vn1,vn1_targ)", np.mean(vn1_targ_err ** 2))
    L.epoch.obs("MSE(vn2,vn2_targ)", np.mean(vn2_targ_err ** 2))
    L.epoch.obs("MSE(vn1, vn2)", np.mean((vn1_pred - vn2_pred) ** 2))
    L.epoch.obs("stdizer reward stddev", stdizer.observed_reward_std)

    line_plots = [
        (
            "epoch mean loss, vn1 & vn2",
            [("vn1", L.epoch_avg.get("loss1")), ("vn2", L.epoch_avg.get("loss2"))],
        ),
        (
            "MSE(vn*, vn*_targ)",
            [
                ("vn1", L.epoch.get("MSE(vn1,vn1_targ)")),
                ("vn2", L.epoch.get("MSE(vn2,vn2_targ)")),
            ],
        ),
        ("MSE(vn1, vn2)", [("MSE", L.epoch.get("MSE(vn1, vn2)"))]),
        (
            "reward stdizer, obs std",
            [("reward std", L.epoch.get("stdizer reward stddev"))],
        ),
        (
            "norm of gradient per layer",
            [(n, L.epoch_avg.get(n)) for n in grad_log_names],
        ),
        (
            "mean: Y, R, pred(vn1,S), pred(vn2,S)",
            [
                ("Y", L.epoch_avg.get("mean Y")),
                ("R", L.epoch_avg.get("mean R")),
                ("vn1", L.epoch_avg.get("mean predict(vn1,S)")),
                ("vn2", L.epoch_avg.get("mean predict(vn2,S)")),
                ("disc pred(S_n)", L.epoch_avg.get("mean pred(vn*,S_n)*d^n")),
            ],
        ),
    ]

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};   last loss vn1 {loss1:.4f}"
        f"\nLR_0={LR_0}, decay={decay}, num_obs_rewards={num_obs_rewards}; tau={tau}; layers={str(layers)} ; sin,cos version"
        f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}"
    )

    plotter.update_plot(im_plots, line_plots, title=title)


# %%
