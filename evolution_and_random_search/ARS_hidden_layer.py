# from numba import njit
import numpy as np

# from numpy import cos, sin, pi, ones, zeros

# from random import sample

import time
import pickle
import os

import datetime

# from ../soft_actor_critic/plot_run_logs import plot_run_logs, elapsed_time_string

from plot_episode_logs import plot_episode_logs, elapsed_time_string


def timestamp():
    return datetime.datetime.now().isoformat().replace(":", "_")


def randn_like(A):
    return np.random.randn(*A.shape)


def augmentedRandomSearch_hiddenLayers(
    env_fn,
    env_name,
    alg_name="unknown_alg",
    directions_per_episode=40,
    episodes_per_epoch=1000,
    epochs=100,
    lr=0.02,
    epoch_plot_fig_handler=None,
    max_steps_per_episode=3000,
    std_dev_exploration_noise=0.01,  # nu
    model=None,
    hidden_size=None,
    observed_state_mean=None,
    observed_state_std=None,
    welford_var_agg=None,
    seed=20,
    min_normalization_std=0.1,
):
    np.random.seed(seed)

    run_params = {
        "directions_per_episode": directions_per_episode,
        "episodes_per_epoch": episodes_per_epoch,
        "epochs": epochs,
        "lr": lr,
        "max_steps_per_episode": max_steps_per_episode,
        "std_dev_exploration_noise": std_dev_exploration_noise,
        "model": model,
        "hidden_size": hidden_size,
        "observed_state_mean=": observed_state_mean,
        "observed_state_std": observed_state_std,
        "seed": seed,
        "min_normalization_std": min_normalization_std,
    }

    start_time = time.time()
    run_datetime_str = timestamp()

    env = env_fn()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # if current_weights is None:
    #     current_weights = np.zeros([action_dim, state_dim])
    if model is None:
        if hidden_size is None:
            hidden_size = int((state_dim + action_dim) / 2)
        current_fc_1 = np.zeros((hidden_size, state_dim))
        current_bias_1 = np.zeros((hidden_size, 1))
        current_fc_2 = np.zeros((action_dim, hidden_size))
    else:
        current_fc_1, current_bias_1, current_fc_2 = model

    if observed_state_mean is None:
        observed_state_mean = np.zeros((state_dim, 1))
    if observed_state_std is None:
        observed_state_std = np.ones((state_dim, 1))

    if welford_var_agg is None:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
        welford_var_agg = np.ones((state_dim, 1))

    num_states_observed = 0

    episode_log = {
        "episode num": [],
        "elapsed time": [],
        "episode time": [],
        "reward": [],
        "reward std": [],
        "episode steps": [],
        "episode done": [],
    }

    _, update_plot = plot_episode_logs(
        episode_log, post_update_fig_handler=epoch_plot_fig_handler,
    )

    def whiten_state(state):
        # print("observed_state_std", observed_state_std)
        return (state - observed_state_mean) / observed_state_std

    def act(state, fc_1, bias_1, fc_2):
        # print("pi_weights, state", pi_weights, state)
        whitened_state = whiten_state(state)
        x = np.matmul(fc_1, whitened_state) + bias_1
        # x = np.maximum(x, 0)  # Relu
        x = np.tanh(x)
        return np.matmul(fc_2, x)

    def env_step_reshape(action):
        state, reward, done, _ = env.step(action)
        return state.reshape(-1, 1), reward, done, _

    def env_reset_reshape():
        state = env.reset()
        return state.reshape(-1, 1)

    def run_trajectory(fc_1, bias_1, fc_2, train=True):
        nonlocal welford_var_agg
        nonlocal observed_state_mean
        nonlocal observed_state_std
        nonlocal num_states_observed
        state = env_reset_reshape()
        total_reward = 0
        for steps in range(max_steps_per_episode):
            num_states_observed += 1

            action = act(state, fc_1, bias_1, fc_2)
            state, reward, done, _ = env_step_reshape(action)

            if abs(state[2, 0]) < 0.001:
                reward = np.array([-100])
                done = True

            total_reward += reward

            if train:
                next_mean = (
                    observed_state_mean
                    + (state - observed_state_mean) / num_states_observed
                )

                welford_var_agg += (state - next_mean) * (state - observed_state_mean)

                observed_state_mean = next_mean

                observed_state_std = np.sqrt(
                    welford_var_agg / num_states_observed
                ).clip(min=min_normalization_std)

            if done:
                break
        if train:
            return total_reward
        else:
            return total_reward, steps, done

    def update_weights(perturbations, reward_pos, reward_neg):
        nonlocal current_fc_1
        nonlocal current_bias_1
        nonlocal current_fc_2

        std_rewards = np.std(np.concatenate([reward_neg, reward_pos]))
        fc_1_adjustment = np.zeros_like(current_fc_1)
        bias_1_adjustment = np.zeros_like(current_bias_1)
        fc_2_adjustment = np.zeros_like(current_fc_2)

        for j, (fc1_1_eps, bias_1_eps, fc1_2_eps) in enumerate(perturbations):
            # weight_adjustment += delta * (reward_pos[j] - reward_neg[j])
            reward_diff = reward_pos[j] - reward_neg[j]

            fc_1_adjustment += fc1_1_eps * reward_diff
            bias_1_adjustment += bias_1_eps * reward_diff
            fc_2_adjustment += fc1_2_eps * reward_diff

        weight_adj_scaler = lr / (std_rewards * directions_per_episode)

        # current_weights += weight_adj_scaler * weight_adjustment
        current_fc_1 += fc_1_adjustment * weight_adj_scaler
        current_bias_1 += bias_1_adjustment * weight_adj_scaler
        current_fc_2 += fc_2_adjustment * weight_adj_scaler

        return std_rewards

    # do iterations

    for episode_num in range(episodes_per_epoch * epochs):
        print(f"episode_num: {episode_num}")
        episode_start_time = time.perf_counter()

        perturbations = [
            (
                randn_like(current_fc_1),
                randn_like(current_bias_1),
                randn_like(current_fc_2),
            )
            for _ in range(directions_per_episode)
        ]

        perturb_pos = [
            (
                current_fc_1 + fc1_1_eps * std_dev_exploration_noise,
                current_bias_1 + bias_1_eps * std_dev_exploration_noise,
                current_fc_2 + fc1_2_eps * std_dev_exploration_noise,
            )
            for (fc1_1_eps, bias_1_eps, fc1_2_eps) in perturbations
        ]

        # perturb_neg = [
        #     (
        #         current_fc_1 - randn_like(current_fc_1) * std_dev_exploration_noise,
        #         current_bias_1 - randn_like(current_bias_1) * std_dev_exploration_noise,
        #         current_fc_2 - randn_like(current_fc_2) * std_dev_exploration_noise,
        #     )
        #     for _ in perturbations
        # ]
        perturb_neg = [
            (
                current_fc_1 - fc1_1_eps * std_dev_exploration_noise,
                current_bias_1 - bias_1_eps * std_dev_exploration_noise,
                current_fc_2 - fc1_2_eps * std_dev_exploration_noise,
            )
            for (fc1_1_eps, bias_1_eps, fc1_2_eps) in perturbations
        ]

        reward_pos = [run_trajectory(*model) for model in perturb_pos]
        reward_neg = [run_trajectory(*model) for model in perturb_neg]

        batch_reward_std = update_weights(perturbations, reward_pos, reward_neg)

        # episode metrics
        current_reward, steps, done = run_trajectory(
            current_fc_1, current_bias_1, current_fc_2, train=False
        )

        episode_log["episode num"].append(episode_num)
        episode_log["elapsed time"].append(time.time() - start_time)
        episode_log["episode time"].append(episode_start_time - start_time)
        episode_log["reward"].append(current_reward)
        episode_log["reward std"].append(batch_reward_std)
        episode_log["episode steps"].append(steps)
        episode_log["episode done"].append(done)

        # End of epoch handling
        if (episode_num + 1) % episodes_per_epoch == 0:
            epoch = (episode_num + 1) // episodes_per_epoch

            elapsed_time = np.round(time.time() - start_time)

            print(
                f"\rEpoch {epoch}; {episode_num} episodes; {episode_num} steps;"
                f"     elapsed time: {elapsed_time_string(elapsed_time)}"
            )

            log_path = (
                f"{os.getcwd()}/model_runs/{alg_name}/{env_name}/{run_datetime_str}"
            )
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_filename = f"/log.pkl"

            with open(log_path + log_filename, "wb") as params_file:
                pickle.dump(
                    {"run params": run_params, "episode log": episode_log}, params_file,
                )

            epoch_str = str(epoch).zfill(len(str(epochs)) + 1)
            model_filename = f"/model_epoch-{epoch_str}.npy"
            with open(log_path + model_filename, "wb") as model_file:
                np.savez(
                    model_file,
                    model=(current_fc_1, current_bias_1, current_fc_2),
                    observed_state_mean=observed_state_mean,
                    observed_state_std=observed_state_std,
                    welford_var_agg=welford_var_agg,
                )

            update_plot()

    return (current_fc_1, current_bias_1, current_fc_2), episode_log, act
