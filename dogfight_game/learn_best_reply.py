import numpy as np
import os
import datetime


import time
import streamlit as st

from dogfight_game import GameEnv, plotGameData, best_reply_game_rollout

st.header("pre-generate a data set")

N_agents = st.number_input(
    label="number of agents", min_value=2, value=10, step=1, format="%.0d"
)

time_steps = st.number_input(
    label=" max time steps", min_value=0, value=10, step=1, format="%.0d",
)

num_states_to_save = st.number_input(
    label="total steps to save", min_value=0, value=10000, step=1, format="%.0d",
)

save_every = st.number_input(
    label="save every X steps", min_value=0, value=1000, step=1, format="%.0d",
)

create_dataset_button = st.button("create dataset")


state_size = 2 * N_agents + N_agents + N_agents  # pos + heading + health
states = np.zeros((state_size, num_states_to_save))
best_actions = np.zeros((2, num_states_to_save))


def timestamp():
    return datetime.datetime.now().isoformat().replace(":", "_")


start_time_stamp = timestamp()


def save_data(states, best_actions):
    save_path = f"{os.getcwd()}/best_reply_lookahead/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = f"{save_path}states_and_actions_{start_time_stamp}.npz"
    with open(filename, "wb") as f:
        np.savez(f, states, best_actions)
    return save_path


if create_dataset_button:
    agent0_2ndReply = True
    agent0_lookahead = True
    start_time = time.time()

    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    X, Y = np.meshgrid(x, y)
    action_options = list(zip(X.ravel(), Y.ravel()))

    i = 0
    next_save = save_every
    while i < num_states_to_save:
        random_seed_initial_conditions = int(time.time() % 1e8)
        random_seed_dynamics = int(time.time() % 1e8)
        env = best_reply_game_rollout(
            N_agents,
            num_states_to_save,
            action_options,
            random_seed_initial_conditions,
            random_seed_dynamics,
            agent0_2ndReply,
            agent0_lookahead,
        )

        i_next = i + env.turnsSoFar

        if i_next < num_states_to_save:
            best_actions[:, i:i_next] = env.saved_actions[:, 0, :]
            states[:, i:i_next] = env.saved_states
        else:
            remaining = num_states_to_save - i
            best_actions[:, i:] = env.saved_actions[:, 0, :][:, :remaining]
            states[:, i:] = env.saved_states[:, :remaining]
        i = i_next
        print(f"{i} states saved so far", end="\r")
        if i > next_save:
            next_save += save_every
            save_data(states, best_actions)

    save_path = save_data(states, best_actions)
    st.write(f"file saved to {save_path}")
