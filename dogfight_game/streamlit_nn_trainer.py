import streamlit as st
from collections import deque
import matplotlib.pyplot as plt

import numpy as np
import torch
import time

from ddqn_agent import Agent

from game_model import GameEnv


def plot_training_info(training_info, n_episodes, st_plot, st_plot_title):
    (
        agent_name,
        round_num,
        i_episode,
        elapsed_time,
        rolling_avg,
        scores,
        scores_rolling_avg,
    ) = training_info

    titleStr = "{}, round {}\nEpisode {}, ({:.2f}s, {:.2f}s/ep)\nAvg Score: {:.2f}".format(
        agent_name,
        round_num,
        i_episode,
        elapsed_time,
        elapsed_time / i_episode,
        rolling_avg,
    )

    print("training_info")
    print(training_info)
    st_plot_title.text(titleStr)

    plt.cla()
    plt.plot(scores)
    plt.plot(scores_rolling_avg)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.xlim(0, n_episodes)  # consistent scale
    # display.clear_output(wait=True)
    st_plot.pyplot(plt.gcf())


def runner(
    env,
    agent,
    agent_name,
    n_episodes=1000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    round_num=0,
):
    t0 = time.time()
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_rolling_avg = []
    steps_in_episode = []

    eps = eps_start  # initialize epsilon
    elapsed_times = []

    print("TRAINING")

    st_plot_title = st.empty()
    st_plot = st.empty()

    for i_episode in range(1, n_episodes + 1):
        print("TRAINING episode: ", str(i_episode))
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        steps_in_episode.append(t)

        # st_steps_this_episode.text("episode lasted {} steps".format(t))
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        rolling_avg = np.mean(scores_window)
        scores_rolling_avg.append(rolling_avg)
        elapsed_times.append(time.time() - t0)

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if i_episode % 10 == 0:
            elapsed_time = time.time() - t0
            training_info = (
                agent_name,
                round_num,
                i_episode,
                elapsed_time,
                rolling_avg,
                scores,
                scores_rolling_avg,
            )
            plot_training_info(training_info, n_episodes, st_plot, st_plot_title)

    state_dict = agent.qnetwork_local.state_dict()
    return (scores, elapsed_times, steps_in_episode, state_dict)


def trainer():
    env = GameEnv(N_agents=8, enemies_stationary=True)
    # env.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    st.write("training using device {}".format(device))
    agentArgs = {
        "device": "cuda:0",
        "state_size": env.getStateDimension(),
        "action_size": 2,
        "seed": 0,
    }
    agent = Agent(**agentArgs)

    runner(env, agent, "ddqn")


if __name__ == "__main__":

    st.write("app started")
    trainer()

    st.write("end of __main__")
