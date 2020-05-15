import gym

import numpy as np
import streamlit as st
import os
import pickle

import datetime


from ES_model_strategy_runner import esRunner

from ARS_optimizer import ARS_optimizer
from CMA_optimizer import CMA_optimizer

from standardizing_linear_agent import StandardizingLinearAgent
from standardizing_mlp_agent import StandardizingMLPAgent, PendulumAgent

from plot_episode_logs import plot_episode_logs
from dogfight_game import GameEnv


def timestamp():
    return datetime.datetime.now().isoformat().replace(":", "_")


# OPTIMIZERS
optimizers_by_name = {"ARS": ARS_optimizer, "CMA": CMA_optimizer}

selected_optim = st.selectbox("Select an optimizer", list(optimizers_by_name))

optimizer_class = optimizers_by_name[selected_optim]

#### AGENTS
agents_by_name = {
    "StandardizingLinearAgent": StandardizingLinearAgent,
    "StandardizingMLPAgent": StandardizingMLPAgent,
    "PendulumAgent": PendulumAgent,
}

selected_agent = st.selectbox("Select an agent", list(agents_by_name))

agent_class = agents_by_name[selected_agent]

#### ENVS
gym_envs = [
    "BipedalWalker-v2",
    "Pendulum-v0",
    "BipedalWalkerHardcore-v2",
    "MountainCarContinuous-v0",
    "LunarLanderContinuous-v2",
]

my_envs = ["dogfight_game"]

selected_env = st.selectbox("Select an environment", gym_envs + my_envs)


run_group_path = (
    f"{os.getcwd()}/model_runs/{selected_optim}/{selected_agent}/{selected_env}/"
)


def run_selector(folder_path="."):
    try:
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
    except FileNotFoundError:
        return None


def examine_existing_model_run(run_dirname):
    st.write(f"You selected run `{run_dirname}`")

    with open(run_dirname + "/log.pkl", "rb") as pickle_file:
        log_info = pickle.load(pickle_file)
    st.write(log_info["run params"])

    episode_log = log_info["episode log"]

    fig, plt_update_fn = plot_episode_logs(episode_log)
    plt_update_fn()
    st.pyplot(fig)

    def checkpoint_selector(folder_path="."):
        filenames = [f for f in sorted(os.listdir(folder_path)) if f[:5] == "model"]
        selected_filename = st.selectbox("Select a file", sorted(filenames))
        return os.path.join(folder_path, selected_filename)

    checkpoint_filename = checkpoint_selector(run_dirname)
    st.write("You selected model checkpoint `%s`" % checkpoint_filename)

    def load_and_run_model(checkpoint_filename, num_runs=3):

        env = gym.make(selected_env)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = agent_class(state_dim, action_dim)

        with open(checkpoint_filename, "rb") as pickle_file:
            agent_state_dict = pickle.load(pickle_file)
        agent.load_agent_state_dict(agent_state_dict)

        for i in range(num_runs):
            print(f"showing {i} of {num_runs} runs")
            state = env.reset()
            for j in range(1000):
                env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if done:
                    break
        env.close()

    run_trained_model = st.button("run trained model (in new window)")
    if run_trained_model:
        load_and_run_model(checkpoint_filename)


run_dirname = run_selector(run_group_path)
if run_dirname is not None:
    examine_existing_model_run(run_dirname)
else:
    st.text("no runs with this config yet")


# train_more = st.button("keep training")
# if train_more:

#     saved_model = np.load(checkpoint_filename)

#     pi_weights = saved_model["weights"]
#     obs_mean = saved_model["observed_state_mean"]
#     obs_std = saved_model["observed_state_std"]

#     st_more_training_plot = st.empty()

#     ac, step_log, episode_log = algorithm(
#         env_fn,
#         env_name=selected_env,
#         episodes_per_epoch=episodes_per_epoch,
#         epochs=epochs,
#         epoch_plot_fig_handler=(lambda fig: st_more_training_plot.pyplot(fig)),
#         directions_per_episode=directions_per_episode,
#         max_steps_per_episode=1000,
#         current_weights=pi_weights,
#         observed_state_mean=obs_mean,
#         observed_state_std=obs_std,
#     )


# Train a fresh model
st.subheader("Train a fresh model")


def env_fn():
    if selected_env in gym_envs:
        return gym.make(selected_env)
    else:
        return GameEnv()


episodes_per_epoch = st.number_input(
    label="episodes per epoch", min_value=0, value=10, step=1, format="%.0d",
)

epochs = st.number_input(label="epochs", min_value=0, value=100, step=1, format="%.0d",)


samples_per_episode = st.number_input(
    label="sample population size per episode",
    min_value=0,
    value=16,
    step=1,
    format="%.0d",
)

random_seed = st.number_input(
    label="random seed", min_value=0, value=1, step=1, format="%.0d",
)

train_fresh_model = st.button("train a fresh model")
fresh_train_plot = st.empty()


def model_trainer_fig_update_handler(fig):
    fresh_train_plot.pyplot(fig)


if train_fresh_model:

    run_datetime_str = timestamp()

    log_path = f"{run_group_path}{run_datetime_str}"

    agent, episode_log = esRunner(
        env_fn,
        env_name=selected_env,
        agent_class=agent_class,
        optimizer_class=optimizer_class,
        # agent_kwargs={},
        optimizer_kwargs={"pop_size": samples_per_episode},
        episodes_per_epoch=episodes_per_epoch,
        epochs=epochs,
        max_steps_per_episode=1000,
        seed=random_seed,
        epoch_plot_fig_handler=model_trainer_fig_update_handler,
        log_path=log_path,
    )
