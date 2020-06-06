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
from damped_spring_noise import dampedSpringNoiseJit
import pendulum_utils as PU


reload(PU)
reload(jnn)
get_ipython().run_line_magic("matplotlib", "inline")
# %%

stdizer = jnn.RewardStandardizer()
for i in range(20):
    print(stdizer.observed_reward_mean, stdizer.observed_reward_std)
    stdizer.observe_reward_vec(randn(50) * 40)


# %%
params = PU.default_pendulum_params


# %%
randKey = lambda: random.PRNGKey(int(100000 * np.random.rand(1)))

# %%
T = 200
traj = PU.pendulumTraj_scan(
    np.pi * (2 * rand() - 1),
    8 * (2 * rand() - 0.5),
    np.clip(
        dampedSpringNoiseJit(100, key=randKey()), -params.max_torque, params.max_torque
    ),
    params,
)
PU.plotTrajInPhaseSpace(traj, params)


# %%
model = jnn.init_network_params_He([2, 80, 80, 1])
# jnn.predict(model,)

# jnn.init_network_params_He


# %%


class Plotter2d(object):
    def __init__(
        self,
        predict,
        model,
        n_grid=100,
        jupyter=True,
        pend_params=PU.default_pendulum_params,
    ):
        self.n_grid = n_grid

        X1, X2 = np.meshgrid(
            np.pi * np.linspace(-1, 1, n_grid), 8 * np.linspace(-1, 1, n_grid)
        )

        X = np.vstack([X1.ravel(), X2.ravel()])
        self.plotX = X
        self.Y = jnn.predict(model, X).reshape(self.n_grid, self.n_grid)

        # fig, (axLoss, axY, axYhat, axResid) = plt.subplots(1, 4, figsize=(8, 8))
        # fig, (axY, axYhat) = plt.subplots(1, 2, figsize=(14, 7))
        fig, (axLoss, axY, axYhat) = plt.subplots(1, 3, figsize=(15, 5))

        self.fig = fig
        self.axY = axY
        self.imY = self.axY.imshow(
            self.Y,
            origin="lower",
            cmap="PiYG",
            alpha=0.9,
            extent=(-np.pi, np.pi, -8, 8),
            aspect=0.5,
        )
        self.axY.set_title("initial NN prediction")
        self.cbarY = self.fig.colorbar(self.imY, ax=self.axY)

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
        THDOT, THDOTDOT = PU.controlledPendulumStep_derivs(
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


# %%
p = Plotter2d(jnn.predict, model, n_grid=100, jupyter=True)
# update_plot(Yhat, loss_list, batch_num)


# %%
# NOTE in this cell, we estimate the average *reward* at each state, just a proof of concept

T = 300
num_epochs = 100
samples_per_epoch = T * 50
batch_size = 100

model = jnn.init_network_params_He([2, 80, 80, 1])

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
                    dampedSpringNoiseJit(T, key=randKey()),
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
        # print(f"episode {e} of {samples_per_epoch/episode_len};"
        # f"   {epoch_memory.shape} samples so far", end="\r")
        e += 1
        theta0 = np.pi * (2 * rand() - 1)
        thetadot0 = 8 * (2 * rand() - 0.5)
        traj = np.array(
            PU.pendulumTraj_scan(
                theta0,
                thetadot0,
                np.clip(
                    dampedSpringNoiseJit(episode_len, key=randKey()),
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


def make_n_step_traj_episode(
    episode_len, n, stdizer, params,
):
    theta0 = np.pi * (2 * rand() - 1)
    thetadot0 = 8 * (2 * rand() - 0.5)
    traj = np.array(
        PU.pendulumTraj_scan(
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
    traj[:, 0] = PU.angle_normalize(traj[:, 0])
    stdizer.observe_reward_vec(traj[:, 2])

    S = traj[0:-n, 0:2]
    R = np.hstack(
        [
            stdizer.standardize_reward(traj[m : -(n - m), 2]).reshape(-1, 1)
            for m in range(n)
        ]
    )
    S_n = traj[n:, 0:2]
    episode = np.hstack([S, S_n, R])
    return episode


make_n_step_traj_episode(5, 1, stdizer, params).T


#%%


# %%# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# NOTE: now trying TD(n)


episode_len = 300
num_epochs = 100
episodes_per_epoch = 10
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3

discount = 0.99

tau = 0.002
LR_0 = 0.00001
decay = 0.99

lookahead = 3

vn1 = jnn.init_network_params_He([2, 80, 80, 1])
vn1_targ = jnn.copy_network(vn1)
vn2 = jnn.init_network_params_He([2, 80, 80, 1])
vn2_targ = jnn.copy_network(vn2)

plotter = Plotter2d(jnn.predict, vn1, n_grid=100, jupyter=True)
stdizer = jnn.RewardStandardizer()


loss_list = []
for i in range(num_epochs):
    epoch_memory = np.zeros((0, 5))
    for j in range(episodes_per_epoch):
        make_n_step_traj_episode(5, lookahead, stdizer, params)
        epoch_memory = np.vstack([epoch_memory, episode])

    sample_order = np.random.permutation(epoch_memory.shape[0])

    LR = LR_0 * decay ** i
    for j in range(int(samples_per_epoch / batch_size)):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = epoch_memory[sample_indices, 0:2]

        S = jnp.transpose(epoch_memory[sample_indices, 0:2])
        S_lookahead = jnp.transpose(epoch_memory[sample_indices, 2:4])
        Y = (discount ** lookahead) * np.minimum(
            jnn.predict(vn1_targ, S_lookahead), jnn.predict(vn2_targ, S_lookahead)
        )
        for t in range(lookahead - 1):
            Y += (discount ** t) * jnp.transpose(
                epoch_memory[sample_indices, (4 + t) : (5 + t)]
            )

        vn1, loss1 = jnn.update_and_loss(vn1, (S, Y), LR)
        loss_list.append(loss1)

        vn2, loss2 = jnn.update_and_loss(vn2, (S, Y), LR)

        if (j + 1) % update_every == 0:
            vn1_targ = jnn.interpolate_networks(vn1_targ, vn1, tau)
            vn2_targ = jnn.interpolate_networks(vn2_targ, vn2, tau)

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};   last loss {loss_list[-1]:.4f}"
        f"\nLR_0={LR_0}, decay={decay}"
    )
    plotter.update_plot(jnn.predict(vn1, plotter.plotX), loss_list, i, title=title)
# %%
# import tree_dynamics_agent as tda
# from importlib import reload

# reload(tda)


# # %%
# tda.Agent(
#     state=np.array([[1], [2]]),
#     dynamics_layer_sizes=[3, 10, 2],
#     reward_layer_sizes=[3, 5, 5, 1],
#     value_layer_sizes=[2, 5, 1],
#     obs_dim=2,
#     act_dim=1,
#     memory_size=100000,
#     batch_size=64,
#     target_num_leaves=20,
# )


# # %%
# delta_S

# rand_state_0 = (0.0, 0.0, sigma, theta, phi, key)

# for i in range(num_episodes):
#     state = env.reset()

#     value_est = Value(state)

#     new_state_node = Node(
#         action=None, state=state, noise_state=rand_state_0, value_est=value_est
#     )
#     T = tree(init_root=new_state_node)

#     for t in range(max_timesteps):
#         while T.num_leaves() < target_num_leaves:
#             state_node = T.softmax_sample_leaf_states()
#             state = state_node.state
#             noise_state = state_node.noise_state
#             noise_state, action = dampedSpringNoiseStep(state.noise_state)
#             state_est = state + DynamicsFn.est(state, action)
#             reward_est = RewardFn.est(state_est, action)
#             value_est = ValueFn.est(state_est)

#             new_state_node = Node(
#                 action=action,
#                 state=state_est,
#                 noise_state=noise_state,
#                 value_est=value_est,
#             )

#             T.add_state(parent=state_node, child=new_state_node, edge_reward=reward_est)

#         target_node = T.softmax_sample_leaf_states()
#         next_node = T.step_toward_and_reroot(target_node)
#         action = next_node.action
#         # NOTE does there need to be a "done" estimator?
#         last_state = state
#         state, reward, done, _ = env.step(action)

#         agent.update(last_state, action, reward, state)
#         agent.learn()
#         DynamicsFn.update(state, last_state, action)
#         RewardFn.update(reward, state, action)
#         ValueFn.update(reward, state)  # ?
