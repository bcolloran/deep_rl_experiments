import numpy as np
import time
import pickle
import os

from plot_episode_logs import plot_episode_logs, elapsed_time_string


def esRunner(
    env_fn,
    env_name,
    agent_class,
    optimizer_class,
    agent_kwargs={},
    optimizer_kwargs={},
    episodes_per_epoch=1000,
    epochs=100,
    epoch_plot_fig_handler=None,
    max_steps_per_episode=3000,
    seed=20,
    log_path=None,
):
    np.random.seed(seed)

    run_params = {
        "env_name": env_name,
        "episodes_per_epoch": episodes_per_epoch,
        "epochs": epochs,
        "seed": seed,
        "agent_kwargs": agent_kwargs,
        "optimizer_kwargs": optimizer_kwargs,
    }

    start_time = time.time()

    env = env_fn()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = agent_class(state_dim, action_dim, **agent_kwargs)
    optimizer = optimizer_class(agent.get_model_param_vector(), **optimizer_kwargs)

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

    for episode_num in range(episodes_per_epoch * epochs):
        print(f"episode_num: {episode_num}")
        episode_start_time = time.perf_counter()

        candidates = optimizer.ask()
        rewards = np.array(
            [agent.run_trajectory_in_env(env, params=c) for c in candidates]
        )
        optimizer.tell(candidates, rewards)

        best_params = optimizer.result[0]
        agent.update_param_vector(best_params)
        test_reward, steps, done = agent.run_trajectory_in_env(env, testing=True)

        episode_log["episode num"].append(episode_num)
        episode_log["elapsed time"].append(time.time() - start_time)
        episode_log["episode time"].append(episode_start_time - start_time)
        episode_log["reward"].append(test_reward)
        episode_log["reward std"].append(np.std(rewards))
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

            if log_path is not None:
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                log_filename = f"/log.pkl"

                with open(log_path + log_filename, "wb") as params_file:
                    pickle.dump(
                        {"run params": run_params, "episode log": episode_log},
                        params_file,
                    )

                epoch_str = str(epoch).zfill(len(str(epochs)) + 1)
                model_filename = f"/model_epoch-{epoch_str}.npy"
                with open(log_path + model_filename, "wb") as model_file:
                    pickle.dump(
                        agent.get_agent_state_dict(), model_file,
                    )

            update_plot()

    return agent, episode_log
