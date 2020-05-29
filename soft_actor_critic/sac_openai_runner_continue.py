import gym

import torch

import numpy as np
import streamlit as st
import os
import pickle

from sac_openai_cuda import sac, OUNoise
from sac_openai_cuda_nets import MLPActorCritic
from plot_run_logs import plot_run_logs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


selected_env = st.selectbox(
    "Select an environment",
    [
        "BipedalWalker-v2",
        "Pendulum-v0",
        "BipedalWalkerHardcore-v2",
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v2",
    ],
)


def env_fn():
    return gym.make(selected_env)


steps_per_epoch = st.number_input(
    label="steps per epoch", min_value=0, value=1000, step=1, format="%.0d",
)

epochs = st.number_input(
    label="epochs", min_value=0, value=1000, step=1, format="%.0d",
)

start_steps = st.number_input(
    label="random steps at start", min_value=0, value=1000, step=1, format="%.0d",
)

seed = st.number_input(label="seed", min_value=0, value=1, step=1, format="%.0d",)


train_fresh_model = st.button("train a fresh model")
fresh_train_plot = st.empty()


def model_trainer_fig_update_handler(fig):
    fresh_train_plot.pyplot(fig)


if train_fresh_model:
    ac, step_log, episode_log = sac(
        env_fn,
        env_name=selected_env,
        start_steps=start_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        update_every=5,
        seed=seed,
        batch_size=128,
        ac_kwargs={"hidden_sizes": (128, 128)},
        lr=0.00007,
        device=device,
        epoch_plot_fig_handler=model_trainer_fig_update_handler,
    )


def run_selector(folder_path="."):
    dir_names = os.listdir(folder_path)
    dirs_by_dir_labels = {
        f"{dirname}"
        f' ({len(os.listdir(f"{folder_path}/{dirname}"))}'
        " checkpoints)": dirname
        for dirname in dir_names
    }
    selected_label = st.selectbox("Select a run", sorted(list(dirs_by_dir_labels)))
    selected_dir = dirs_by_dir_labels[selected_label]
    return os.path.join(folder_path, selected_dir)


run_dirname = run_selector(f"model_runs/{selected_env}")
st.write(f"You selected run `{run_dirname}`")


with open(run_dirname + "/log.pkl", "rb") as pickle_file:
    # print(pickle_file)
    log_info = pickle.load(pickle_file)
# print(list(log_info))
st.write(log_info["run params"])

episode_log = log_info["episode log"]
step_log = log_info["step log"]

fig, plt_update_fn = plot_run_logs(episode_log, step_log)
plt_update_fn()
st.pyplot(fig)


def checkpoint_selector(folder_path="."):
    filenames = [f for f in sorted(os.listdir(folder_path)) if f[:5] == "model"]
    selected_filename = st.selectbox("Select a file", sorted(filenames))
    return os.path.join(folder_path, selected_filename)


checkpoint_filename = checkpoint_selector(run_dirname)
st.write("You selected model checkpoint `%s`" % checkpoint_filename)


def load_and_run_trained_net(checkpoint_filename, device):

    env = gym.make(selected_env)

    actor_net = MLPActorCritic(env.observation_space, env.action_space)

    actor_net.load_state_dict(torch.load(checkpoint_filename))
    actor_net.to(device)

    # act_dim = env.action_space.shape[0]
    # noise_process = OUNoise(act_dim, seed=2040)
    # print(actor_net)
    for i in range(10):
        print(i)
        state = env.reset()
        for j in range(300):
            # action = agent.act(state)
            env.render()
            action = actor_net.act(
                torch.as_tensor(state, dtype=torch.float32).to(device),
                # deterministic=True
            )
            # action = np.clip(action + noise_process.sample(), -1, 1)
            # action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


run_trained_model = st.button("run trained model (in new window)")
if run_trained_model:
    load_and_run_trained_net(checkpoint_filename, device)


def newAcFromOld(old_net):
    def constructor(*args, **kwargs):
        new_net = MLPActorCritic(*args, **kwargs)
        new_net.pi = old_net.pi
        new_net.q1 = old_net.q1
        new_net.q2 = old_net.q2
        new_net.device = old_net.device
        return new_net

    return constructor


train_more = st.button("keep training")
if train_more:
    env = gym.make("BipedalWalker-v2")
    actor_net = MLPActorCritic(env.observation_space, env.action_space)

    actor_net.load_state_dict(torch.load(checkpoint_filename))
    actor_net.to(device)

    st_more_training_plot = st.empty()

    actor_net2 = sac(
        env_fn,
        actor_critic=newAcFromOld(actor_net),
        start_steps=0,
        steps_per_epoch=1000,
        # max_ep_len=100,
        update_every=1,
        epochs=1000,
        use_logger=False,
        alpha=0.2,
        lr=0.0003,
        # batch_size=256,
        # lr=7e-3,
        add_noise=True,
        device=device,
        st_plot_fn=st_more_training_plot.pyplot,
    )
