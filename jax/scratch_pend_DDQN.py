# %%
import numpy as np
from numpy.random import randn, rand

import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as jnp

from collections import OrderedDict as odict

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

params = PU.default_pendulum_params

# %%# %%
# NOTE now let's try to estimate the VALUE function for an agent using the policy:
# "always take a spring-noise-random action"
# NOTE: now trying TD(n)
# NOTE: now trying cos,sin state representation

episode_len = 200
num_epochs = 1000
episodes_per_epoch = 10
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3

greed_eps_0 = 1
greed_eps_min = 0.05
greed_eps_decay = 0.99


def greed_eps(t):
    return max(greed_eps_min, greed_eps_0 * greed_eps_decay ** t)


discount = 0.97

tau = 0.002  # best: ~0.005?

LR_0 = 0.00003  # best ~1e-5?
decay = 0.993

num_lookahead = 4

layers = [4, 64, 64, 1]
qn1 = jnn.init_network_params_He(layers)
qn1_targ = jnn.copy_network(qn1)
# qn2 = jnn.init_network_params_He(layers)
# qn2_targ = jnn.copy_network(qn2)

plotter = PP.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotX = np.vstack([np.cos(plotter.plotX1), np.sin(plotter.plotX1), plotter.plotX2])
stdizer = jnn.RewardStandardizer()

L = SL.Logger()

grad_log_names = []
for l_num in range(len(layers) - 1):
    for l_type in ["w", "b"]:
        grad_log_names.append(f"qn1_d{l_type}{l_num}")

for i in range(num_epochs):
    epoch_memory = None
    for j in range(episodes_per_epoch):
        th0 = np.pi * (2 * rand() - 1)
        thdot0 = 8 * (2 * rand() - 0.5)
        St, Action, Cost = PU.pendulumTrajDDQN(
            th0, thdot0, episode_len, qn1, greed_eps(i)
        )
        L.epoch_avg.obs(f"trajectory cost", np.sum(Cost))

        mem = PU.make_n_step_data_from_DDQN(
            St, Action, Cost, num_lookahead, stdizer, params
        )

        if epoch_memo:
            # if random.uniform(subkey) > eps:
            #     u = ddqnBestAction(Q1, np.cos(th), np.sin(th), thdot)
            # else:
            #     index = random.randint(randkey, (1,), 0, len(U_opts))[0]
            #     u = U_opts[index]ry is None:
            epoch_memory = mem
        else:
            epoch_memory = np.hstack([epoch_memory, mem])

    sample_order = np.random.permutation(epoch_memory.shape[1])
    LR = LR_0 * decay ** i
    for j in range(int(np.floor(len(sample_order) / batch_size))):
        sample_indices = sample_order[(j * batch_size) : ((j + 1) * batch_size)]

        batch = epoch_memory[:, sample_indices]

        SA_0 = batch[0:4, :]

        cos_th_n = batch[4, :]
        sin_th_n = batch[6, :]
        thdot_n = batch[7, :]

        A_n = PU.ddqnBestActionsVect(qn1, cos_th_n, sin_th_n, thdot_n)
        SA_n = np.vstack([batch[4:7, :], A_n])

        Y = (discount ** num_lookahead) * jnn.predict(qn1_targ, SA_n)

        L.epoch_avg.obs(f"mean pred(qn1_targ,SA_n)*d^n", np.mean(Y))

        R = np.zeros_like(Y)
        for t in range(num_lookahead):
            R += (discount ** t) * batch[(8 + t) : (9 + t), :]
        Y += R

        L.epoch_avg.obs(f"mean Y", np.mean(Y))
        L.epoch_avg.obs(f"mean R", np.mean(R))
        L.epoch_avg.obs(f"mean predict(qn1,S)", np.mean(jnn.predict(qn1, SA_0)))

        qn1, loss1, norm_grads1 = jnn.update_with_loss_and_norm_grad(qn1, (SA_0, Y), LR)
        L.epoch_avg.obs(f"q loss1", loss1)
        for l_num, (dw, db) in enumerate(norm_grads1):
            L.epoch_avg.obs(f"qn1_dw{l_num}", dw)
            L.epoch_avg.obs(f"qn1_db{l_num}", db)

        if (j + 1) % update_every == 0:
            qn1_targ = jnn.interpolate_networks(qn1_targ, qn1, tau)

    im_plots = []

    for torque in [-2, 0, 2]:
        im_plots.append(
            (
                f"qn1 pred, torque={torque}",
                jnn.predict(qn1, np.vstack([plotX, torque * np.ones(plotX.shape[1])])),
            )
        )

    im_plots.append(
        (
            f"qn1 pred, (torque=2) - (torque=-2)",
            jnn.predict(qn1, np.vstack([plotX, 2 * np.ones(plotX.shape[1])]))
            - jnn.predict(qn1, np.vstack([plotX, -2 * np.ones(plotX.shape[1])])),
        )
    )

    L.end_epoch()
    L.epoch.obs("stdizer reward stddev", stdizer.observed_reward_std)
    L.epoch.obs("greedy eps", greed_eps(i))

    line_plots = [
        ("trajectory cost", [("cost", L.epoch_avg.get("trajectory cost"))],),
        ("epoch mean loss, qn1", [("qn1", L.epoch_avg.get("q loss1"))],),
        (
            "norm of gradient per layer - q",
            [(n, L.epoch_avg.get(n)) for n in grad_log_names],
        ),
        (
            "mean: Y, R, pred(qn1,S)",
            [
                ("Y", L.epoch_avg.get("mean Y")),
                ("R", L.epoch_avg.get("mean R")),
                ("qn1", L.epoch_avg.get("mean predict(qn1,S)")),
                # ("qn2", L.epoch_avg.get("mean predict(qn2,S)")),
                ("disc pred(S_n)", L.epoch_avg.get("mean pred(qn1_targ,SA_n)*d^n")),
            ],
        ),
        (
            "reward stdizer, obs std",
            [("reward std", L.epoch.get("stdizer reward stddev"))],
            {"yscale": "linear"},
        ),
        ("greedy eps", [("eps", L.epoch.get("greedy eps"))], {"yscale": "linear"}),
        ("sample traj inputs", [("inputs", Action)], {"yscale": "linear"}),
    ]

    num_traj_to_plot = 1
    traj_plots = [("sample traj", [("traj", PU.angle_normalize(St[0, :]), St[1, :])],)]

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};   last loss vn1 {loss1:.4f}"
        f"\nLR_0={LR_0}, decay={decay}, num_lookahead={num_lookahead}; tau={tau}; layers={str(layers)} ; sin,cos version"
        f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}"
    )

    plotter.update_plot(im_plots, line_plots, traj_plots, title=title)


# %%

T = 5
theta0 = np.pi * (2 * rand() - 1)
thetadot0 = 8 * (2 * rand() - 0.5)
state_and_params = (theta0, thetadot0, PU.default_pendulum_params)
U = []
for t in range(T):
    print(t)
    minQ = np.inf
    for torque in np.linspace(-2, 2, 9):
        qhat = jnn.predict(
            qn1,
            jnp.array(
                [
                    [np.cos(state_and_params[0])],
                    [np.sin(state_and_params[0])],
                    [state_and_params[1]],
                    [torque],
                ]
            ),
        )
        if qhat.item() < minQ:
            minQ = qhat
            bestU = torque
        print(torque, qhat.item())
    print(f"minQ {minQ}, bestU {bestU}")
    U.append(bestU)
    state_and_params, state_and_cost = PU.controlledPendulumStep_scan(
        state_and_params, bestU
    )

traj = PU.pendulumTraj_scan(theta0, thetadot0, np.array(U))
PP.plotTrajInPhaseSpace(traj, PU.default_pendulum_params, U)

# %%

layers = [4, 64, 64, 1]
qn1 = jnn.init_network_params_He(layers)

U_opts = np.linspace(-params.max_torque, params.max_torque, 7)

th = 0.4
thdot = 4

state_action_template = np.vstack(
    [np.ones_like(U_opts), np.ones_like(U_opts), np.ones_like(U_opts), U_opts]
)
s_a = state_action_template * np.array([[np.cos(th), np.sin(th), thdot, 1]]).T

np.argmax(jnn.predict(qn1, s_a))
# %%

th0 = np.pi * (2 * rand() - 1)
thdot0 = 8 * (2 * rand() - 0.5)
St, Ac, Cost = PU.pendulumTrajDDQN(th0, thdot0, 100, qn1)

Mem = PU.make_n_step_data_from_DDQN(St, Ac, Cost, 3, stdizer, params)
print(Mem[:, :6])
# %%
L.epoch_avg.get("mean predict(qn1,S)")


# %%
