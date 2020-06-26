# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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

from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
from noise_procs import dampedSpringNoise
import pendulum_utils as PU

reload(PU)
reload(jnn)
get_ipython().run_line_magic("matplotlib", "inline")

# %%
params = PU.default_pendulum_params
Plotter2d = PU.PendulumValuePlotter
randKey = jnn.randKey
randKey()

# %%
T = 200
traj = PU.pendulumTraj_scan(
    np.pi * (2 * rand() - 1),
    8 * (2 * rand() - 0.5),
    np.clip(
        dampedSpringNoise(100, key=randKey()), -params.max_torque, params.max_torque
    ),
    params,
)
PU.plotTrajInPhaseSpace(traj, params)


# %%
model = jnn.init_network_params_He([2, 80, 80, 1])

# %%
# this chunk plots the true expected value at each state
n_grid = 50

X1, X2 = np.meshgrid(np.pi * np.linspace(-1, 1, n_grid), 8 * np.linspace(-1, 1, n_grid))
th = X1.ravel()
thdot = X2.ravel()
Y = PU.cost_of_state(th, thdot, params, 0).reshape(n_grid, n_grid)

fig, axY = plt.subplots(1, 1, figsize=(15, 5))

imY = axY.imshow(
    Y,
    origin="lower",
    cmap="PiYG",
    alpha=0.9,
    extent=(-np.pi, np.pi, -8, 8),
    aspect=0.5,
)
axY.set_title("true RAW expected value of state")
fig.colorbar(imY, ax=axY)

# %%
Y = PU.cost_of_state(th, thdot, params, 0)
stdzr = jnn.RewardStandardizer()
stdzr.observe_reward_vec(Y)
Ystd = stdzr.standardize_reward(Y).reshape(n_grid, n_grid)


fig, axY = plt.subplots(1, 1, figsize=(15, 5))

imY = axY.imshow(
    Ystd,
    origin="lower",
    cmap="PiYG",
    alpha=0.9,
    extent=(-np.pi, np.pi, -8, 8),
    aspect=0.5,
)
axY.set_title("true STANDARDIZED expected value of state")
fig.colorbar(imY, ax=axY)

# %%

p = Plotter2d(jnn.predict, model, n_grid=100, jupyter=True)

# update_plot(Yhat, loss_list, batch_num)


# %%
# NOTE in this cell, we estimate the average *reward* at each state, just a proof of concept

T = 300
num_epochs = 1000
samples_per_epoch = T * 50
batch_size = 100

model = jnn.init_network_params_He([2, 100, 100, 1])

plotter = Plotter2d(jnn.predict, model, n_grid=100, jupyter=True)
stdizer = jnn.RewardStandardizer()

loss_list = []
for i in range(num_epochs):
    epoch_memory = np.zeros((0, 3))

    while epoch_memory.shape[0] < samples_per_epoch:
        traj = np.array(
            PU.pendulumTraj_scan(
                np.pi * (2 * rand() - 1),
                8 * (2 * rand() - 0.5),
                np.clip(
                    dampedSpringNoise(T, key=randKey()),
                    -params.max_torque,
                    params.max_torque,
                ),
                params,
            )
        )
        traj[:, 0] = PU.angle_normalize(traj[:, 0])
        epoch_memory = np.vstack([epoch_memory, traj])

    sample_order = np.random.permutation(samples_per_epoch)
    j = 0
    while j < samples_per_epoch:
        sample_indices = sample_order[j : j + batch_size]
        j += batch_size
        X = epoch_memory[sample_indices, 0:2].T
        for r in epoch_memory[sample_indices, 2]:
            stdizer.observe_reward(r)
        Y = stdizer.standardize_reward(epoch_memory[sample_indices, 2].T)
        model, loss = jnn.update_and_loss(model, (X, Y), LR=0.0005)
        loss_list.append(loss)

    plotter.update_plot(jnn.predict(model, plotter.plotX), loss_list, i)
# %%
stdzr.observed_reward_std, stdizer.observed_reward_std


# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# we'll try semi-gradient TD(0), sutton page 203


episode_len = 300
num_epochs = 100
samples_per_epoch = episode_len * 10
batch_size = 100
update_every = 3

discount = 0.99

tau = 0.002
LR_0 = 0.00001
decay = 0.99

vn1 = jnn.init_network_params_He([2, 80, 80, 1])
vn1_targ = jnn.copy_network(vn1)
vn2 = jnn.init_network_params_He([2, 80, 80, 1])
vn2_targ = jnn.copy_network(vn2)

plotter = Plotter2d(jnn.predict, vn1, n_grid=100, jupyter=True)
stdizer = jnn.RewardStandardizer()


loss_list = []
for i in range(num_epochs):
    epoch_memory = np.zeros((0, 5))

    e = 0
    while epoch_memory.shape[0] < samples_per_epoch:
        e += 1
        theta0 = np.pi * (2 * rand() - 1)
        thetadot0 = 8 * (2 * rand() - 0.5)
        traj = np.array(
            PU.pendulumTraj_scan(
                theta0,
                thetadot0,
                np.clip(
                    dampedSpringNoise(episode_len, key=randKey()),
                    -params.max_torque,
                    params.max_torque,
                ),
                params,
            )
        )
        traj[:, 0] = PU.angle_normalize(traj[:, 0])
        # for r in :
        stdizer.observe_reward_vec(traj[:, 2])
        R = stdizer.standardize_reward(traj[:-1, 2]).reshape(-1, 1)
        S = traj[:-1, 0:2]
        S_next = traj[1:, 0:2]

        episode = np.hstack([S, S_next, R])
        epoch_memory = np.vstack([epoch_memory, episode])

    sample_order = np.random.permutation(samples_per_epoch)
    # j = 0
    # while j < samples_per_epoch:
    # sample_indices = sample_order[j : (j + batch_size)]
    # j += batch_size
    LR = LR_0 * decay ** i
    for j in range(int(samples_per_epoch / batch_size)):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = epoch_memory[sample_indices, 0:2]

        S = jnp.transpose(epoch_memory[sample_indices, 0:2])
        S_next = jnp.transpose(epoch_memory[sample_indices, 2:4])
        R = jnp.transpose(epoch_memory[sample_indices, 4:5])

        Y = R + discount * np.minimum(
            jnn.predict(vn1_targ, S_next), jnn.predict(vn2_targ, S_next)
        )

        vn1, loss1 = jnn.update_and_loss(vn1, (S, Y), LR)
        loss_list.append(loss1)
        vn2, loss2 = jnn.update_and_loss(vn2, (S, Y), LR)

        if (j + 1) % update_every == 0:
            vn1_targ = jnn.interpolate_networks(vn1_targ, vn1, tau)
            vn2_targ = jnn.interpolate_networks(vn2_targ, vn2, tau)

    title = (
        f"epoch {i} of {num_epochs}; LR={LR:.6f}; last loss {loss_list[-1]:.4f}"
        f"\nLR_0={LR_0}, decay={decay}"
    )
    plotter.update_plot(jnn.predict(vn1, plotter.plotX), loss_list, i, title=title)

# %%


make_n_step_traj_episode(5, 1, stdizer, params).T


# %%# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# NOTE: now trying TD(n)


episode_len = 300
num_epochs = 10000
episodes_per_epoch = 10
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3

discount = 0.99

tau = 0.002
LR_0 = 0.00001
decay = 0.993

lookahead = 4

vn1 = jnn.init_network_params_He([2, 80, 80, 1])
vn1_targ = jnn.copy_network(vn1)
vn2 = jnn.init_network_params_He([2, 80, 80, 1])
vn2_targ = jnn.copy_network(vn2)

plotter = Plotter2d(jnn.predict, vn1, n_grid=100, jupyter=True)
stdizer = jnn.RewardStandardizer()


loss_list = []
for i in range(num_epochs):
    epoch_memory = None  # np.zeros((0, 4+lookahead))
    for j in range(episodes_per_epoch):
        episode = PU.make_n_step_traj_episode(episode_len, lookahead, stdizer, params)
        if epoch_memory is None:
            epoch_memory = episode
        else:
            epoch_memory = np.vstack([epoch_memory, episode])

    sample_order = np.random.permutation(epoch_memory.shape[0])

    LR = LR_0 * decay ** i
    for j in range(int(samples_per_epoch / batch_size)):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = jnp.transpose(epoch_memory[sample_indices, :])

        S = batch[0:2, :]
        S_lookahead = batch[2:4, :]

        Y = (discount ** lookahead) * np.minimum(
            jnn.predict(vn1_targ, S_lookahead), jnn.predict(vn2_targ, S_lookahead)
        )
        for t in range(lookahead - 1):
            Y += (discount ** t) * batch[(4 + t) : (4 + 1 + t), :]

        vn1, loss1 = jnn.update_and_loss(vn1, (S, Y), LR)
        loss_list.append(loss1)

        vn2, loss2 = jnn.update_and_loss(vn2, (S, Y), LR)

        if (j + 1) % update_every == 0:
            vn1_targ = jnn.interpolate_networks(vn1_targ, vn1, tau)
            vn2_targ = jnn.interpolate_networks(vn2_targ, vn2, tau)

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};   last loss {loss_list[-1]:.4f}"
        f"\nLR_0={LR_0}, decay={decay}, lookahead={lookahead}"
    )
    plotter.update_plot(jnn.predict(vn1, plotter.plotX), loss_list, i, title=title)

# %%
