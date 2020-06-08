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
from damped_spring_noise import dampedSpringNoiseJit
import pendulum_utils as PU


reload(PU)
reload(jnn)


get_ipython().run_line_magic("matplotlib", "inline")

params = PU.default_pendulum_params

# %%# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# NOTE: now trying TD(n)
# NOTE: now trying cos,sin state representation

episode_len = 300
num_epochs = 500
episodes_per_epoch = 10
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3

discount = 0.99

tau = 0.005
LR_0 = 0.00001
decay = 0.993

lookahead = 4

vn1 = jnn.init_network_params_He([3, 80, 80, 1])
vn1_targ = jnn.copy_network(vn1)
vn2 = jnn.init_network_params_He([3, 80, 80, 1])
vn2_targ = jnn.copy_network(vn2)

plotter = PU.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotX = np.vstack([np.cos(plotter.plotX1), np.sin(plotter.plotX1), plotter.plotX2])
stdizer = jnn.RewardStandardizer()

loss1_list_epoch = []
loss2_list_epoch = []
reward_std_epoch = []
mse_targ1_list_epoch = []
mse_targ2_list_epoch = []
mse_vn1_vn2_list_epoch = []

for i in range(num_epochs):
    epoch_memory = None
    for j in range(episodes_per_epoch):
        episode = PU.make_n_step_sin_cos_traj_episode(
            episode_len, lookahead, stdizer, params
        )
        if epoch_memory is None:
            epoch_memory = episode
        else:
            epoch_memory = np.vstack([epoch_memory, episode])

    sample_order = np.random.permutation(epoch_memory.shape[0])

    loss1_list_episode = []
    loss2_list_episode = []
    LR = LR_0 * decay ** i
    for j in range(int(samples_per_epoch / batch_size)):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = jnp.transpose(epoch_memory[sample_indices, :])

        S = batch[0:3, :]

        S_lookahead = batch[3:6, :]

        Y = (discount ** lookahead) * np.minimum(
            jnn.predict(vn1_targ, S_lookahead), jnn.predict(vn2_targ, S_lookahead)
        )
        for t in range(lookahead - 1):
            Y += (discount ** t) * batch[(6 + t) : (7 + t), :]

        vn1, loss1 = jnn.update_and_loss(vn1, (S, Y), LR)
        loss1_list_episode.append(loss1)

        vn2, loss2 = jnn.update_and_loss(vn2, (S, Y), LR)
        loss2_list_episode.append(loss2)

        if (j + 1) % update_every == 0:
            vn1_targ = jnn.interpolate_networks(vn1_targ, vn1, tau)
            vn2_targ = jnn.interpolate_networks(vn2_targ, vn2, tau)

    vn1_pred = jnn.predict(vn1, plotX)
    vn2_pred = jnn.predict(vn2, plotX)
    vn1_targ_pred = jnn.predict(vn1_targ, plotX)
    vn2_targ_pred = jnn.predict(vn2_targ, plotX)

    vn1_targ_err = vn1_pred - vn1_targ_pred
    vn2_targ_err = vn2_pred - vn2_targ_pred

    loss1_list_epoch.append(np.mean(loss1_list_episode))
    loss2_list_epoch.append(np.mean(loss2_list_episode))
    mse_targ1_list_epoch.append(np.mean(vn1_targ_err ** 2))
    mse_targ2_list_epoch.append(np.mean(vn2_targ_err ** 2))
    mse_vn1_vn2_list_epoch.append(np.mean((vn1_pred - vn2_pred) ** 2))
    reward_std_epoch.append(stdizer.observed_reward_std)

    im_plots = [
        ("vn1 prediction", vn1_pred),
        ("vn2 prediction", vn2_pred),
        ("vn1_targ prediction", vn1_targ_pred),
        ("vn2_targ prediction", vn2_targ_pred),
        ("vn1 - vn1_targ", vn1_targ_err),
        ("vn2 - vn2_targ", vn2_targ_err),
    ]

    line_plots = [
        ("epoch mean loss", [("vn1", loss1_list_epoch), ("vn2", loss2_list_epoch)]),
        (
            "MSE(vn*, vn*_targ)",
            [("vn1", mse_targ1_list_epoch), ("vn2", mse_targ2_list_epoch),],
        ),
        ("MSE(vn1, vn2)", [("mse", mse_vn1_vn2_list_epoch)],),
        ("reward std", [("reward std"), reward_std_epoch]),
    ]

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};   last loss vn1 {loss1:.4f}"
        f"\nLR_0={LR_0}, decay={decay}, lookahead={lookahead}; tau={tau}, sin,cos version"
    )

    plotter.update_plot(im_plots, line_plots, title=title)


# %%
