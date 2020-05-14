import numpy as np

import time
import streamlit as st

# from game_model import doTurn, randomActions, intitialState, GameEnv
# from game_model import GameEnv
# from plot_game_data import plotGameData

from dogfight_game import GameEnv, plotGameData

randn = np.random.randn
rand = np.random.rand

print("\n\n\n\n\n++++++++++++++++++++NEW RUN++++++++++++++")


# start = time.time()
# initial_state = intitialState()
# positions, headings, health, hits, reward = doTurn(
# initial_state, randomActions(initial_state[0].shape[1]))
# end = time.time()
# print("Elapsed = %s" % (end - start))

# start = time.time()
# initial_state = intitialState()
# positions, headings, health, hits, reward = doTurn(
# initial_state, randomActions(initial_state[0].shape[1]))
# end = time.time()
# print("Elapsed 2nd run = %s" % (end - start))


N_agents = st.number_input(
    label="number of agents", min_value=2, value=10, step=1, format="%.0d"
)

time_steps = st.number_input(
    label="time steps", min_value=0, value=10, step=1, format="%.0d",
)


random_seed_initial_conditions = st.number_input(
    label="random seed for initial conditions",
    min_value=0,
    value=1,
    step=1,
    format="%.0d",
)


np.random.seed(random_seed_initial_conditions)
env = GameEnv(N_agents=N_agents, enemy_type="straight")


x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
X, Y = np.meshgrid(x, y)
action_options = list(zip(X.ravel(), Y.ravel()))

start = time.time()

random_seed_dynamics = st.number_input(
    label="random seed for dynamics", min_value=0, value=1, step=1, format="%.0d"
)

np.random.seed(random_seed_dynamics)
for i in range(time_steps):

    default_actions = env.pickDefaultActions()
    actions = np.zeros_like(default_actions)
    for agent in range(actions.shape[1]):
        action, bestReward = env.pickBestAgentRewardsForActions(
            agent, action_options, default_actions
        )

        actions[:, agent] = action

    action, bestReward = env.pickBestAgentRewardsForActions(0, action_options, actions)
    # distance, yaw = agent_action[0], agent_action[1]
    # print(action)
    actions[:, 0] = action

    next_state, reward, done, _ = env.step_actions_for_all(actions)
    if i == 0:
        print("time to complete first step (jit): %s" % (time.time() - start))
    if done:
        st.write("done! iter: ", i)
        st.write("rewards:\n", reward)
        if not np.isnan(env.getDeathTimes()[0]):
            st.write("Main agent died")
        else:
            st.write(f"Agent0 killed all others at turn {i}")
        break
    # st.write(reward)


print("time to complete all steps: %s" % (time.time() - start))

positions, headings, health, hits, reward = env.getTurnDataTup()
start = time.time()
st.pyplot(plotGameData(positions, hits, health, env.getDeathTimes()))
print("time to draw plot: %s" % (time.time() - start))

st.write(f"enemies still alive: {np.isnan(env.getDeathTimes()).sum()}")
st.line_chart(env.rewards[0, :])
