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


def plot_run_logs(
    episode_log, step_log, total_steps=None, post_update_fig_handler=None
):
    if total_steps is None:
        total_steps = len(step_log["step time"])
    (
        figure,
        (ax_reward_vs_episode, ax_steps_vs_episode, ax_reward_vs_step, ax_time_vs_step),
    ) = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
    ax_reward_vs_episode.set_xlabel("episode")
    ax_reward_vs_episode.set_ylabel("reward")

    ax_steps_vs_episode.set_xlabel("episode")
    ax_steps_vs_episode.set_ylabel("steps")

    ax_reward_vs_step.set_xlim([0, total_steps])
    ax_reward_vs_step.set_xlabel("step")
    ax_reward_vs_step.set_ylabel("reward")

    ax_time_vs_step.set_xlim([0, total_steps])
    ax_time_vs_step.set_xlabel("step")
    ax_time_vs_step.set_ylabel("time")
    ax_time_vs_step.set_yscale("log")

    (l_step,) = ax_time_vs_step.plot([], [], ".", color="k")
    l_step.set_label("step")
    (l_act,) = ax_time_vs_step.plot([], [], ".", color="b")
    l_act.set_label("act")
    (l_train,) = ax_time_vs_step.plot([], [], ".", color="g")
    l_train.set_label("train")
    (l_env,) = ax_time_vs_step.plot([], [], ".", color="r")
    l_env.set_label("env")

    l_step.set_label("step")
    l_act.set_label("act")
    l_train.set_label("train")
    l_env.set_label("env")
    ax_time_vs_step.legend(
        bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.0
    )
    plt.tight_layout()

    def update_plot():
        num_steps = len(step_log["reward"])

        elapsed_time = step_log["elapsed time"][-1]

        episode_num = len(episode_log["episode num"])

        plot_title = (
            f"episode {episode_num};  step {num_steps}"
            f"\nlast 100 avg episode score: {np.mean(episode_log['reward'][-100:]):.2f}"
            f"\ntime: {elapsed_time_string(elapsed_time)}"
            f" {elapsed_time / num_steps:.4f}s/step)"
        )

        ax_reward_vs_episode.set_title(plot_title)
        ax_reward_vs_episode.plot(episode_log["reward"], "k")

        ax_steps_vs_episode.plot(
            episode_log["episode num"], episode_log["episode steps"], "k"
        )

        ax_reward_vs_step.plot(step_log["reward"], "k")

        x = np.arange(num_steps)
        l_step.set_data(x, step_log["step time"])
        l_act.set_data(x, step_log["act time"])
        l_train.set_data(x, step_log["train time"])
        l_env.set_data(x, step_log["env time"])

        if post_update_fig_handler is not None:
            post_update_fig_handler(figure)

        return figure

    return figure, update_plot
