import numpy as np

import time
import streamlit as st

from dogfight_game import GameEnv, plotGameData, best_reply_game_rollout

print("\n\n\n\n\n++++++++++++++++++++NEW RUN++++++++++++++")


N_agents = st.number_input(
    label="number of agents", min_value=2, value=10, step=1, format="%.0d"
)

time_steps = st.number_input(
    label="time steps", min_value=0, value=10, step=1, format="%.0d",
)


random_seed_initial_conditions = st.number_input(
    label="random seed for initial conditions",
    min_value=1,
    value=1,
    step=1,
    format="%.0d",
)
np.random.seed(random_seed_initial_conditions)


x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
X, Y = np.meshgrid(x, y)
action_options = list(zip(X.ravel(), Y.ravel()))

random_seed_dynamics = st.number_input(
    label="random seed for dynamics", min_value=1, value=1, step=1, format="%.0d"
)


agent0_2ndReply = st.checkbox("Agent 0 best reply to other agents best replies")
agent0_lookahead = st.checkbox("Agent 0 looks ahead one turn")


start_time = time.time()
env = best_reply_game_rollout(
    N_agents,
    time_steps,
    action_options,
    random_seed_initial_conditions,
    random_seed_dynamics,
    agent0_2ndReply,
    agent0_lookahead,
)
print("time to complete rollouts: %s" % (time.time() - start_time))

st.write("agent 0 reward:\n", np.sum(env.rewards[0, :]))


positions, headings, health, hits, reward = env.getTurnDataTup()
start_time = time.time()
st.pyplot(plotGameData(positions, hits, health, env.getDeathTimes()))
print("time to draw plot: %s" % (time.time() - start_time))

st.write(f"enemies still alive: {np.isnan(env.getDeathTimes()).sum()}")


st.header("create a data set")

create_dataset_button = st.button("create dataset")

time_steps = st.number_input(
    label="total steps", min_value=0, value=1000000, step=1, format="%.0d",
)
