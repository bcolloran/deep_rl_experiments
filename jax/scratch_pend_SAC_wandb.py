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

import wandb

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

run = wandb.init(project="SAC pendulum")

wandb.config.episode_len = 100
wandb.config.num_epochs = 1000
wandb.config.episodes_per_epoch = 300
wandb.config.num_random_episodes = 1000
wandb.config.samples_per_epoch = (
    wandb.config.episode_len * wandb.config.episodes_per_epoch
)
wandb.config.batch_size = 100
wandb.config.update_every = 3
wandb.config.memory_size = int(1e6)
# int(samples_per_epoch / batch_size)

wandb.config.td_steps = 5
wandb.config.discount = 0.99

wandb.config.seed = 0

wandb.config.LR = 3 * 1e-4
wandb.config.tau = 0.005


agent = SAC.Agent(
    obs_dim=3,
    act_dim=1,
    action_max=2,
    memory_size=wandb.config.memory_size,
    batch_size=wandb.config.batch_size,
    td_steps=wandb.config.td_steps,
    discount=wandb.config.discount,
    LR=wandb.config.LR,
    tau=wandb.config.tau,
    seed=wandb.config.seed,
    state_transformer=PU.expand_state_cos_sin,
)

wandb.config.LRdecay = 0.99
LR_0 = agent.LR


plotter = PP.PendulumValuePlotter2(n_grid=100, jupyter=False, panel_size=9)
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
    T=wandb.config.episode_len,
    dynamics_fn=dynamics_fn,
    random_action_fn=random_action_fn,
    S0_fn=PU.random_initial_state,
    noise0_fn=DSN.dampedSpringNoiseStateInit,
)

for i in range(wandb.config.num_random_episodes):
    key = DSN.next_key(key)
    S, A, R = random_episode_fn(key)
    D = np.zeros_like(R)
    agent.remember_episode(S, A, R, D)
    print(f"\rrandom episode {i} of {wandb.config.num_random_episodes}", end="\r")

policy_episode_fn = jtu.make_agent_policynet_episode_fn(
    T=wandb.config.episode_len,
    policy_net_fn=agent.make_policynet_act_fn(),
    dynamics_fn=PU.make_dyn_fn(),
    noise_fn=DSN.normalStep,
    S0_fn=PU.random_initial_state,
    noise0_fn=DSN.normalNoiseInit,
)

t0 = time.time()

for i in range(wandb.config.num_epochs):
    LR = LR_0 * wandb.config.LRdecay ** i
    for j in range(wandb.config.episodes_per_epoch):
        print(
            f"\rstarting epoch {i} of {wandb.config.num_epochs} (episode {j}/{wandb.config.episodes_per_epoch})",
            end="\r",
        )
        key = DSN.next_key(key)
        S, A, R = policy_episode_fn(agent.pi, key)
        D = np.zeros_like(R)
        agent.remember_episode(S, A, R, D)
        q_loss_val, pi_loss_val, alpha_loss_val = agent.update_2()

        # A_pred, log_prob = agent.predict_action_and_log_prob(S, agent.new_eps(A.shape))
        # L.epoch_avg.obs(f"mean predict(qn2,S)", np.mean(jnn.predict(qn2, S)))

        wandb.log(
            {
                f"episode mean reward": np.mean(R).item(),
                f"episode mean prediction q(S,A)": np.mean(
                    agent.predict_q(S, A)
                ).item(),
                f"q loss": q_loss_val.item(),
                f"pi loss": pi_loss_val.item(),
                f"alpha (entropy weight)": agent.alpha.item(),
            }
        )

    wandb.log(
        {
            "approx GPU mem": 32
            * sum(
                x.size for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue)
            ),
            "stdizer reward stddev": agent.reward_standardizer.observed_reward_std.item(),
            "Learning rate": LR,
        }
    )

    im_plots = [(f"pi pred", agent.predict_pi_opt(plotX), {"diverging": True})]
    traj_plots = [("most recent trajectory", [(f"traj", S[:, 0], S[:, 1])])]

    title = f"final trajectory, epoch {i}"

    fig = plotter.update_plot(im_plots, None, traj_plots, title=title)
    wandb.log({"final trajectory of epoch": wandb.Image(fig)})

run.finish()

# im_plots = []

# for torque in [-2, 0, 2]:
#     plotX
#     plotA = torque * np.ones((plotX.shape[0], 1))
#     im_plots.append((f"qn1 pred, torque={torque}", agent.predict_q(plotX, plotA)))

# im_plots.append(
#     (
#         f"qn1 pred, (torque=2) - (torque=-2)",
#         agent.predict_q(plotX, 2 * np.ones((plotX.shape[0], 1)))
#         - agent.predict_q(plotX, -2 * np.ones((plotX.shape[0], 1))),
#     )
# )
# im_plots.append((f"pi pred", agent.predict_pi_opt(plotX), {"diverging": True}))

# # num_traj_to_plot = 1
# traj_plots = [("most recent trajectory", [(f"traj", S[:, 0], S[:, 1])])]

# title = (
#     f"epoch {i}/{num_epochs};   LR={LR:.9f};     elapsed mins: {(time.time()-t0)/60:.2f}"
#     f"\nLR_0={LR_0:.7f}, decay={LRdecay}; td_steps={td_steps}; agent version"
#     f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}"
# )

# plotter.update_plot(im_plots, line_plots, traj_plots, title=title)


# %%

# show the agent's behavior in the gym
import gym

env = gym.make("Pendulum-v0")

episode_len = 400

state = env.reset()
act_fn_needs_noise = agent.make_agent_act_fn()
noise_state = DSN.mvNormalNoiseInit(key)

for t in range(episode_len):
    env.render()
    noise_state, eps = DSN.mvNormalStep(noise_state, 1)
    state_th_thdot = (np.arccos(state[0]), state[2])

    action = act_fn_needs_noise(state_th_thdot, eps)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break
env.close()
