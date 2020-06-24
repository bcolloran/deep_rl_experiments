# %%

import numpy as np
from numpy.random import randn, rand
from functools import partial

import time
import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as jnp
from jax.experimental.stax import serial, parallel, Relu, Dense, FanOut

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import OrderedDict as odict

from jax.config import config

config.update("jax_debug_nans", True)

from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
import jax_trajectory_utils as jtu
import damped_spring_noise as DSN
import pendulum_utils as PU
import pendulum_plotting as PP
import simple_logger as SL
import SAC_agent as SAC

import replay_buffer as RB

reload(PU)
reload(RB)
reload(PP)
reload(SL)
reload(SAC)
reload(DSN)
reload(jnn)
reload(jtu)

get_ipython().run_line_magic("matplotlib", "inline")

params = PU.default_pendulum_params

# %%

episode_len = 10
num_epochs = 3
episodes_per_epoch = 3
num_random_episodes = 3
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 10
update_every = 3
memory_size = int(1e6)
# int(samples_per_epoch / batch_size)

td_steps = 4
discount = 0.99

agent = SAC.Agent(
    obs_dim=3,
    act_dim=1,
    action_max=2,
    memory_size=memory_size,
    batch_size=batch_size,
    td_steps=td_steps,
    discount=discount,
    LR=3 * 1e-4,
    tau=0.005,
    update_interval=1,
    grad_steps_per_update=1,
    seed=0,
    state_transformer=PU.expand_state_cos_sin,
)

LRdecay = 0.99
LR_0 = agent.LR

dynamics_fn = jit(lambda S, A: PU.pendulum_step(S, A, params))
policy_fn = jit(
    lambda S, eps: SAC.action(agent.pi, PU.expand_state_cos_sin(S), eps, agent.pi_fn)
)

plotter = PP.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotX = plotter.plotX.T
L = SL.Logger()

key = random.PRNGKey(seed=0)

for i in range(num_random_episodes):
    S0 = PU.random_initial_state()
    noise0 = DSN.dampedSpringNoiseStateInit()

    S, A, R = jtu.make_random_episode(
        episode_len, dynamics_fn, DSN.dampedSpringNoiseStep, S0, noise0
    )
    agent.remember_episode(S, A, R)
    key, _ = random.split(key)
    print(f"\rrandom episode {i} of {num_random_episodes}", end="\r")

for i in range(num_epochs):

    LR = LR_0 * LRdecay ** i
    for j in range(episodes_per_epoch):
        print(
            f"\rstarting epoch {i} of {num_epochs} (episode {j}/{episodes_per_epoch})",
            end="\r",
        )
        key, _ = random.split(key)
        pol_dyn_scan_fn = jtu.make_scan_policy_dynamics_noise_step_fn(
            dynamics_fn, policy_fn, DSN.normalStep  # agent.make_agent_act_fn(),
        )
        S0 = PU.random_initial_state()

        S, A, R = jtu.make_agent_episode_noisy(episode_len, S0, key, pol_dyn_scan_fn)
        agent.remember_episode(S, A, R)
        q_loss_val, pi_loss_val, alpha_loss_val = agent.update()

        #     L.epoch_avg.obs(f"mean pred(qn*,S_n)*d^n", np.mean(Y))
        #     L.epoch_avg.obs(f"mean Y", np.mean(Y))
        L.epoch_avg.obs(f"mean R", np.mean(R))
        L.epoch_avg.obs(f"mean predict(q,S,A)", np.mean(agent.predict_q(S, A)))
        #     L.epoch_avg.obs(f"mean predict(qn2,S)", np.mean(jnn.predict(qn2, S)))
        L.epoch_avg.obs(f"q loss", q_loss_val)
        L.epoch_avg.obs(f"pi loss", pi_loss_val)
        L.epoch_avg.obs(f"alpha", agent.alpha)
    #     for l_num, (dw, db) in enumerate(norm_grads1):
    #         L.epoch_avg.obs(f"qn1_dw{l_num}", dw)
    #         L.epoch_avg.obs(f"qn1_db{l_num}", db)
    #     L.epoch_avg.obs(f"q loss2", loss2)
    #     for l_num, (dw, db) in enumerate(norm_grads2):
    #         L.epoch_avg.obs(f"qn2_dw{l_num}", dw)
    #         L.epoch_avg.obs(f"qn2_db{l_num}", db)

    im_plots = []

    for torque in [-2, 0, 2]:
        plotX
        plotA = torque * np.ones((plotX.shape[0], 1))
        im_plots.append((f"qn1 pred, torque={torque}", agent.predict_q(plotX, plotA)))

    # im_plots.append(
    #     (
    #         f"qn1 pred, (torque=2) - (torque=-2)",
    #         jnn.predict(qn1, np.vstack([plotX, 2 * np.ones(plotX.shape[1])]))
    #         - jnn.predict(qn1, np.vstack([plotX, -2 * np.ones(plotX.shape[1])])),
    #     )
    # )

    L.end_epoch()
    L.epoch.obs("stdizer reward stddev", agent.reward_standardizer.observed_reward_std)

    line_plots = [
        (
            "reward stdizer, obs std",
            [("reward std", L.epoch.get("stdizer reward stddev"))],
        ),
        (
            "epoch mean loss, q",
            [
                ("q", L.epoch_avg.get("q loss")),
                # ("qn2", L.epoch_avg.get("q loss2"))
            ],
        ),
        #     (
        #         "norm of gradient per layer - q",
        #         [(n, L.epoch_avg.get(n)) for n in grad_log_names],
        #     ),
        (
            "mean: Y, R, pred(qn1,S), pred(qn2,S)",
            [
                # ("Y", L.epoch_avg.get("mean Y")),
                ("R", L.epoch_avg.get("mean R")),
                ("q_pred", L.epoch_avg.get("mean predict(q,S,A)")),
                # ("qn2", L.epoch_avg.get("mean predict(qn2,S)")),
                # ("disc pred(S_n)", L.epoch_avg.get("mean pred(qn*,S_n)*d^n")),
            ],
        ),
        ("alpha", [("alpha", L.epoch_avg.get("alpha"))]),
        ("most recent actions", [("A", A.flatten(), {"yscale": "linear"})],),
    ]

    # num_traj_to_plot = 1
    traj_plots = [("most recent trajectory", [(f"traj", S[:, 0], S[:, 1])])]

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};"
        f"\nLR_0={LR_0:.7f}, decay={LRdecay}; td_steps={td_steps}; agent version"
        f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}"
    )

    plotter.update_plot(im_plots, line_plots, traj_plots, title=title)


# %%

