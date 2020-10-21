# %%

import numpy as np
import time
import jax

from jax import grad, random, jit
import jax.numpy as jnp
import gc

from jax.config import config

config.update("jax_debug_nans", True)

from IPython import get_ipython
from IPython.display import display, clear_output

from importlib import reload

import jax_nn_utils as jnn
import jax_trajectory_utils_2 as jtu
import noise_procs as DSN

import pendulum_plotting as PP

import simple_logger as SL
import SAC_agent as SAC

import replay_buffer as RB

import gym

reload(RB)
reload(SL)
reload(SAC)
reload(DSN)
reload(jnn)
reload(jtu)

get_ipython().run_line_magic("matplotlib", "inline")

# %%
# env = gym.make("BipedalWalker-v3")
# env = gym.make("LunarLanderContinuous-v2")

# %%  PENDULUM SETTINGS

env = gym.make("Pendulum-v0")

max_episode_len = 100
num_epochs = 1000
episodes_per_epoch = 300
num_random_episodes = 1000

action_max = 2


# %%

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

td_steps = 3

discount = 0.99

batch_size = 100
update_every = 3
memory_size = int(1e6)
seed = 100


agent = SAC.Agent(
    obs_dim=state_dim,
    act_dim=action_dim,
    action_max=action_max,
    memory_size=memory_size,
    batch_size=batch_size,
    td_steps=td_steps,
    discount=discount,
    LR=3 * 1e-4,
    tau=0.005,
    update_interval=update_every,
    grad_steps_per_update=1,
    seed=seed,
    state_transformer=lambda S: S,
)

# act_fn_needs_noise = agent.make_agent_act_fn()

policynet_act_fn = agent.make_policynet_act_fn()

# %
# RUN WITH RANDOM PARAMS AND SPRING DAMPED NOISE
# TO SEED MEMORY AND START LEARNING REWARD NORMALIZATION


key = random.PRNGKey(seed=seed)
noise_state = DSN.dampedSpringNoiseStateInit(key, dim=action_dim)

for episode in range(num_random_episodes):
    state = env.reset()
    S = np.zeros((max_episode_len, state_dim))
    A = np.zeros((max_episode_len, action_dim))
    R = np.zeros((max_episode_len, 1))
    D = np.zeros((max_episode_len, 1))
    for t in range(max_episode_len):
        # env.render()
        noise_state, action = DSN.dampedSpringNoiseStep(noise_state)
        next_state, reward, done, _ = env.step(action)
        S[t] = state
        A[t] = action
        R[t] = reward
        D[t] = done
        state = next_state
        if done:
            break
    agent.remember_episode(S[: t + 1], A[: t + 1], R[: t + 1], D[: t + 1])
    print(f"\repisode {episode} done", end="\r")


env.close()


# %%

noise_state = DSN.mvNormalNoiseInit(key)

LRdecay = 0.99
LR_0 = agent.LR

plotter = PP.PendulumValuePlotter2(jupyter=True)
L = SL.Logger()

key = random.PRNGKey(seed=0)


t0 = time.time()
for epoch in range(num_epochs):
    for episode in range(episodes_per_epoch):
        state = env.reset()
        S = np.zeros((max_episode_len, state_dim))
        A = np.zeros((max_episode_len, action_dim))
        R = np.zeros((max_episode_len, 1))
        D = np.zeros((max_episode_len, 1))
        for t in range(max_episode_len):
            noise_state, eps = DSN.mvNormalStep(noise_state, action_dim)

            action = policynet_act_fn(agent.pi, state, eps)

            next_state, reward, done, _ = env.step(action)
            S[t] = state
            A[t] = action
            R[t] = reward
            D[t] = done
            state = next_state
            if done:
                break

        S, A, R, D = S[: t + 1], A[: t + 1], R[: t + 1], D[: t + 1]
        agent.remember_episode(S, A, R, D)

        L.epoch_avg.obs(f"mean R", np.mean(R))
        L.epoch_avg.obs(f"episode length", t)

        # TRAIN!
        # for u in range(t):
        q_loss_val, pi_loss_val, alpha_loss_val = agent.update_2()
        L.epoch_avg.obs(f"q loss", q_loss_val)
        L.epoch_avg.obs(f"pi loss", pi_loss_val)
        L.epoch_avg.obs(f"alpha loss", alpha_loss_val)

        L.epoch_avg.obs(f"mean predict(q,S,A)", np.mean(agent.predict_q(S, A)))
        L.epoch_avg.obs(f"alpha", agent.alpha)

        print(f"\repoch {epoch}, episode {episode} done", end="\r")

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
        ("epoch avg loss, q", [("q", L.epoch_avg.get("q loss"))]),
        ("epoch avg loss, pi", [("pi", L.epoch_avg.get("pi loss"))]),
        ("epoch avg loss, alpha", [("alpha", L.epoch_avg.get("alpha loss"))]),
        (
            "epoch mean episode length",
            [("t", L.epoch_avg.get("episode length"))],
            {"yscale": "linear"},
        ),
        (
            "mean: Y, R, pred(qn1,S), pred(qn2,S)",
            [
                ("R", L.epoch_avg.get("mean R")),
                ("q(S,A)", L.epoch_avg.get("mean predict(q,S,A)")),
            ],
        ),
        ("alpha", [("alpha", L.epoch_avg.get("alpha"))]),
        (
            "approx GPU mem",
            [("bytes", L.epoch.get("approx GPU mem"))],
            {"yscale": "linear"},
        ),
    ]

    title = (
        f"epoch {epoch}/{num_epochs}; elapsed mins: {(time.time()-t0)/60:.2f}"
        f"\nLR_0={LR_0:.7f}, decay={LRdecay}; td_steps={td_steps}; agent version"
        f"\nupdate_every={update_every}, discount={discount}, batch_size={batch_size}; episodes_per_epoch={episodes_per_epoch}; max_episode_len={max_episode_len}"
    )

    plotter.update_plot(None, line_plots, None, title=title)

    # show the agent's current behavior
    state = env.reset()
    for t in range(max_episode_len):
        env.render()
        noise_state, eps = DSN.mvNormalStep(noise_state, action_dim)
        action = policynet_act_fn(agent.pi, state, eps)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

env.close()

# %%

import gym

state = env.reset()
act_fn_needs_noise = agent.make_agent_act_fn()
noise_state = DSN.mvNormalNoiseInit(key)

for t in range(max_episode_len):
    env.render()
    noise_state, eps = DSN.mvNormalStep(noise_state, 1)

    action = act_fn_needs_noise(state, eps)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break
env.close()
