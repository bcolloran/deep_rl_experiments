import gym

import numpy as np
import streamlit as st
import os
import pickle

from ARS_bc import augmentedRandomSearch
from ARS_hidden_layer import augmentedRandomSearch_hiddenLayers

# from sac_openai_cuda import sac, OUNoise
# from sac_openai_cuda_nets import MLPActorCritic
from plot_episode_logs import plot_episode_logs
from dogfight_game import GameEnv


algs_by_name = {
    "ARS": augmentedRandomSearch,
    "ARS_hidden_layer": augmentedRandomSearch_hiddenLayers,
}

selected_alg = st.selectbox("Select an algorithm", list(algs_by_name))

algorithm = algs_by_name[selected_alg]

gym_envs = [
    "BipedalWalker-v2",
    "Pendulum-v0",
    "BipedalWalkerHardcore-v2",
    "MountainCarContinuous-v0",
    "LunarLanderContinuous-v2",
]

my_envs = ["dogfight_game"]

selected_env = st.selectbox("Select an environment", gym_envs + my_envs)


def env_fn():
    if selected_env in gym_envs:
        return gym.make(selected_env)
    else:
        return GameEnv()


episodes_per_epoch = st.number_input(
    label="episodes per epoch", min_value=0, value=10, step=1, format="%.0d",
)

epochs = st.number_input(label="epochs", min_value=0, value=100, step=1, format="%.0d",)


directions_per_episode = st.number_input(
    label="perturbations per episode", min_value=0, value=16, step=1, format="%.0d",
)

random_seed = st.number_input(
    label="random seed", min_value=0, value=1, step=1, format="%.0d",
)

# start_steps = st.number_input(
#     label="random steps at start", min_value=0, value=10000, step=1, format="%.0d",
# )


train_fresh_model = st.button("train a fresh model")
fresh_train_plot = st.empty()


def model_trainer_fig_update_handler(fig):
    fresh_train_plot.pyplot(fig)


if train_fresh_model:
    ac, step_log, episode_log = algorithm(
        env_fn,
        env_name=selected_env,
        alg_name=selected_alg,
        episodes_per_epoch=episodes_per_epoch,
        epochs=epochs,
        epoch_plot_fig_handler=model_trainer_fig_update_handler,
        directions_per_episode=directions_per_episode,
        max_steps_per_episode=1000,
        seed=random_seed,
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


run_dirname = run_selector(f"model_runs/{selected_alg}/{selected_env}")
st.write(f"You selected run `{run_dirname}`")


with open(run_dirname + "/log.pkl", "rb") as pickle_file:
    # print(pickle_file)
    log_info = pickle.load(pickle_file)
# print(list(log_info))
st.write(log_info["run params"])

episode_log = log_info["episode log"]
# step_log = log_info["step log"]

fig, plt_update_fn = plot_episode_logs(episode_log)
plt_update_fn()
st.pyplot(fig)


def checkpoint_selector(folder_path="."):
    filenames = [f for f in sorted(os.listdir(folder_path)) if f[:5] == "model"]
    selected_filename = st.selectbox("Select a file", sorted(filenames))
    return os.path.join(folder_path, selected_filename)


checkpoint_filename = checkpoint_selector(run_dirname)
st.write("You selected model checkpoint `%s`" % checkpoint_filename)


def load_and_run_model(checkpoint_filename, num_runs=4):

    env = gym.make(selected_env)

    saved_model = np.load(checkpoint_filename)
    print("saved_model", saved_model)

    pi_weights = saved_model["weights"]
    obs_mean = saved_model["observed_state_mean"]
    obs_std = saved_model["observed_state_std"]

    def normalize(inputs):
        return (inputs - obs_mean) / obs_std

    def act(state, pi_weights):
        return np.matmul(pi_weights, normalize(state.reshape(-1, 1)))

    for i in range(num_runs):
        print(f"showing {i} of {num_runs} runs")
        state = env.reset()
        for j in range(300):
            env.render()
            action = act(state, pi_weights)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


run_trained_model = st.button("run trained model (in new window)")
if run_trained_model:
    load_and_run_model(checkpoint_filename)


train_more = st.button("keep training")
if train_more:

    saved_model = np.load(checkpoint_filename)

    pi_weights = saved_model["weights"]
    obs_mean = saved_model["observed_state_mean"]
    obs_std = saved_model["observed_state_std"]

    st_more_training_plot = st.empty()

    ac, step_log, episode_log = algorithm(
        env_fn,
        env_name=selected_env,
        episodes_per_epoch=episodes_per_epoch,
        epochs=epochs,
        epoch_plot_fig_handler=(lambda fig: st_more_training_plot.pyplot(fig)),
        directions_per_episode=directions_per_episode,
        max_steps_per_episode=1000,
        current_weights=pi_weights,
        observed_state_mean=obs_mean,
        observed_state_std=obs_std,
    )
