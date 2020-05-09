import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def elapsed_time_string(elapsed_time):
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} sec"
    elif elapsed_time < 60 * 60:
        time_str = f"{elapsed_time/60:.2f} min"
    else:
        time_str = f"{elapsed_time/(60*60):.4f} hr"
    return time_str


# def rolling
#     mylist = [1, 2, 3, 4, 5, 6, 7]
#     N = 3
#     cumsum, moving_aves = [0], []

#     for i, x in enumerate(mylist, 1):
#         cumsum.append(cumsum[i-1] + x)
#         if i>=N:
#             moving_ave = (cumsum[i] - cumsum[i-N])/N
#             #can do stuff with moving_ave here
#             moving_aves.append(moving_ave)


def plot_episode_logs(episode_log, post_update_fig_handler=None):

    (
        figure,
        (ax_reward_vs_episode, ax_steps_vs_episode, ax_batch_reward_std_vs_episode),
    ) = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    ax_reward_vs_episode.set_xlabel("episode")
    ax_reward_vs_episode.set_ylabel("reward")

    ax_steps_vs_episode.set_xlabel("episode")
    ax_steps_vs_episode.set_ylabel("steps")

    ax_batch_reward_std_vs_episode.set_xlabel("episode")
    ax_batch_reward_std_vs_episode.set_ylabel("std dev of rewards")

    plt.tight_layout()

    def update_plot():

        elapsed_time = episode_log["elapsed time"][-1]

        num_episode = len(episode_log["episode num"])

        plot_title = (
            f"episode {num_episode}"
            f"\nlast 100 avg episode score: {np.mean(episode_log['reward'][-100:]):.2f}"
            f"\ntime: {elapsed_time_string(elapsed_time)}"
            f" {elapsed_time / num_episode:.4f}s/episode)"
        )

        ax_reward_vs_episode.set_title(plot_title)
        ax_reward_vs_episode.plot(episode_log["reward"], "k")

        ax_steps_vs_episode.plot(
            episode_log["episode num"], episode_log["episode steps"], "k"
        )

        ax_batch_reward_std_vs_episode.plot(
            episode_log["episode num"], episode_log["reward std"], "k"
        )

        if post_update_fig_handler is not None:
            post_update_fig_handler(figure)

        return figure

    return figure, update_plot
