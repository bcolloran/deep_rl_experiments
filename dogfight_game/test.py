
import numpy as np

import time
import streamlit as st

# from game_model import doTurn, randomActions, intitialState, GameEnv
from game_model import GameEnv
from plot_game_data import plotGameData

np.random.seed(123)

randn = np.random.randn
rand = np.random.rand


print("\n\n\n\n\n++++++++++++++++++++NEW RUN++++++++++++++")


# start = time.time()
# initial_state = intitialState()
# positions, headings, health, hits, reward = doTurn(initial_state, randomActions(initial_state[0].shape[1]))
# end = time.time()
# print("Elapsed = %s" % (end - start))

# start = time.time()
# initial_state = intitialState()
# positions, headings, health, hits, reward = doTurn(initial_state, randomActions(initial_state[0].shape[1]))
# end = time.time()
# print("Elapsed 2nd run = %s" % (end - start))

x = np.linspace(-1,1,5)
y = np.linspace(-1,1,5)
X, Y = np.meshgrid(x, y)
actionsSample = list(zip(X.ravel(),Y.ravel()))


start = time.time()
env = GameEnv(N_agents=99, enemy_type="straight")
for i in range(300):
    # action = (rand(2)*2)-1
    action, bestReward = env.pickBestAgent0RewardsForActions(actionsSample)
    # if bestReward == 0.0:
    #     action = (rand(2)*2)-1

    next_state, reward, done, _ = env.step(np.array(action))
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
st.line_chart(env.rewards[0,:])

