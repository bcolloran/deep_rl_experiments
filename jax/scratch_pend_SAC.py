# %%

import numpy as np
import time
import jax
from jax import grad, random, jit
import jax.numpy as jnp
import gc

# from jax.config import config
# config.update("jax_debug_nans", True)

from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
import jax_trajectory_utils_2 as jtu
import noise_procs as DSN
import pendulum_utils_2 as PU
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

episode_len = 100
num_epochs = 1000
episodes_per_epoch = 300
num_random_episodes = 1000
samples_per_epoch = episode_len * episodes_per_epoch
batch_size = 100
update_every = 3
memory_size = int(1e6)
# int(samples_per_epoch / batch_size)

td_steps = 3
discount = 0.99

seed = 0

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
    seed=seed,
    state_transformer=PU.expand_state_cos_sin,
)

LRdecay = 0.99
LR_0 = agent.LR


plotter = PP.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotX = plotter.plotX.T
L = SL.Logger()

key = random.PRNGKey(seed=0)

# with jax.disable_jit():
dynamics_fn = PU.make_dyn_fn()


@jit
def random_action_fn(noise_state):
    noise_state, eps = DSN.dampedSpringNoiseStep(noise_state)
    return noise_state, jnp.clip(eps, -params.max_torque, params.max_torque)


random_episode_fn = jtu.make_random_episode_fn(
    T=episode_len,
    dynamics_fn=dynamics_fn,
    random_action_fn=random_action_fn,
    S0_fn=PU.random_initial_state,
    noise0_fn=DSN.dampedSpringNoiseStateInit,
)

for i in range(num_random_episodes):
    key = DSN.next_key(key)
    S, A, R = random_episode_fn(key)
    agent.remember_episode(S, A, R)
    print(f"\rrandom episode {i} of {num_random_episodes}", end="\r")

policy_episode_fn = jtu.make_agent_policynet_episode_fn(
    T=episode_len,
    policy_net_fn=agent.make_policynet_act_fn(),
    dynamics_fn=PU.make_dyn_fn(),
    noise_fn=DSN.normalStep,
    S0_fn=PU.random_initial_state,
    noise0_fn=DSN.normalNoiseInit,
)

t0 = time.time()

for i in range(num_epochs):
    LR = LR_0 * LRdecay ** i
    for j in range(episodes_per_epoch):
        print(
            f"\rstarting epoch {i} of {num_epochs} (episode {j}/{episodes_per_epoch})",
            end="\r",
        )
        key = DSN.next_key(key)
        S, A, R = policy_episode_fn(agent.pi, key)
        agent.remember_episode(S, A, R)
        q_loss_val, pi_loss_val, alpha_loss_val = agent.update_2()

        # L.episode.obs(f"q loss", q_loss_val)
        # L.episode.obs(f"pi loss", pi_loss_val)
        # L.episode.obs(f"mean R", np.mean(R))
        # L.episode.obs(f"mean predict(q,S,A)", np.mean(agent.predict_q(S, A)))

        # A_pred, log_prob = agent.predict_action_and_log_prob(S, agent.new_eps(A.shape))

        L.epoch_avg.obs(f"mean R", np.mean(R))
        L.epoch_avg.obs(f"mean predict(q,S,A)", np.mean(agent.predict_q(S, A)))
        L.epoch_avg.obs(f"q loss", q_loss_val)
        L.epoch_avg.obs(f"pi loss", pi_loss_val)
        L.epoch_avg.obs(f"alpha", agent.alpha)

        L.epoch_avg.obs(f"mean R", np.mean(R))
        # L.epoch_avg.obs(f"mean predict(q,S,A)", np.mean(agent.predict_q(S, A)))
        # L.epoch_avg.obs(f"mean predict(qn2,S)", np.mean(jnn.predict(qn2, S)))

    im_plots = []

    for torque in [-2, 0, 2]:
        plotX
        plotA = torque * np.ones((plotX.shape[0], 1))
        im_plots.append((f"qn1 pred, torque={torque}", agent.predict_q(plotX, plotA)))

    im_plots.append(
        (
            f"qn1 pred, (torque=2) - (torque=-2)",
            agent.predict_q(plotX, 2 * np.ones((plotX.shape[0], 1)))
            - agent.predict_q(plotX, -2 * np.ones((plotX.shape[0], 1))),
        )
    )
    im_plots.append((f"pi pred", agent.predict_pi_opt(plotX), {"diverging": True}))

    L.end_epoch()
    L.epoch.obs("stdizer reward stddev", agent.reward_standardizer.observed_reward_std)

    L.epoch.obs(
        "approx GPU mem",
        32
        * sum(x.size for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue)),
    )

    line_plots = [
        (
            "reward stdizer, obs std",
            [("reward std", L.epoch.get("stdizer reward stddev"))],
        ),
        # ("episode q loss", [("q", L.episode.get("q loss"))],),
        # ("episode pi loss", [("pi", L.episode.get("pi loss"))],),
        (
            "epoch mean loss, q",
            [
                ("q", L.epoch_avg.get("q loss")),
                # ("qn2", L.epoch_avg.get("q loss2"))
            ],
        ),
        ("epoch avg pi loss", [("pi", L.epoch_avg.get("pi loss"))]),
        #     (
        #         "norm of gradient per layer - q",
        #         [(n, L.epoch_avg.get(n)) for n in grad_log_names],
        #     ),
        (
            "mean: Y, R, pred(qn1,S), pred(qn2,S)",
            [
                # ("Y", L.epoch_avg.get("mean Y")),
                ("R", L.epoch_avg.get("mean R")),
                ("q(S,A)", L.epoch_avg.get("mean predict(q,S,A)")),
                # ("qn2", L.epoch_avg.get("mean predict(qn2,S)")),
                # ("disc pred(S_n)", L.epoch_avg.get("mean pred(qn*,S_n)*d^n")),
            ],
        ),
        ("alpha", [("alpha", L.epoch_avg.get("alpha"))]),
        (
            "most recent episode",
            [
                ("A", A.flatten()),
                ("sin(S[:,0])", np.sin(S[:, 0].flatten())),
                ("cos(S[:,0])", np.cos(S[:, 0].flatten())),
                ("S[:,1]", S[:, 1].flatten()),
            ],
            {"yscale": "linear"},
        ),
        (
            "approx GPU mem",
            [("bytes", L.epoch.get("approx GPU mem"))],
            {"yscale": "linear"},
        ),
    ]

    # num_traj_to_plot = 1
    traj_plots = [("most recent trajectory", [(f"traj", S[:, 0], S[:, 1])])]

    title = (
        f"epoch {i}/{num_epochs};   LR={LR:.9f};     elapsed mins: {(time.time()-t0)/60:.2f}"
        f"\nLR_0={LR_0:.7f}, decay={LRdecay}; td_steps={td_steps}; agent version"
        f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}"
    )

    plotter.update_plot(im_plots, line_plots, traj_plots, title=title)


# %%

dvals = [x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue)]
dvals = sorted(dvals, key=lambda x: x.size, reverse=True)

list(map(np.shape, dvals))
list(map(type, dvals))

dvals_const = [
    x
    for x in gc.get_objects()
    if isinstance(x, jax.xla.DeviceValue) and isinstance(x, jax.xla.DeviceConstant)
]

dvals_const

dvals_del = [
    x
    for x in gc.get_objects()
    if isinstance(x, jax.xla.DeviceValue) and x._check_if_deleted()
]

list(map(np.shape, dvals))
list(map(lambda x: x.dtype, dvals))
dvals[3].size
list(map(lambda x: x.size, dvals))

32 * sum(x.size for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue))

# %%

