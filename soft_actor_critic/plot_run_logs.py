import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess


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


def plot_run_logs(episode_log, step_log, total_steps=None):
    if total_steps is None:
        total_steps = len(step_log["step time"])
    figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")

    ax2.set_xlim([0, total_steps])
    ax2.set_xlabel("step")
    ax2.set_ylabel("reward")

    ax3.set_xlim([0, total_steps])
    ax3.set_xlabel("step")
    ax3.set_ylabel("time")
    ax3.set_yscale("log")

    # l_step, _ = ax3.plot(step_log["step time"])
    # l_step.set_label("step")
    # l_act, _ = ax3.plot(step_log["act time"])
    # l_act.set_label("act")
    # l_train, _ = ax3.plot(step_log["train time"])
    # l_train.set_label("train")
    # l_env, _ = ax3.plot(step_log["env time"])
    # l_env.set_label("env")
    first_plot = True

    def update_plot(first_plot, elapsed_time=None):
        num_steps = len(step_log["reward"])

        if elapsed_time is None:
            elapsed_time = step_log["elapsed time"][-1]

        episode_num = len(episode_log["episode num"])

        plot_title = (
            f"episode {episode_num};  step {num_steps}"
            f"\last 100 avg episode score: {np.mean(episode_log['reward'][-100:]):.2f}"
            f"\ntime: {elapsed_time_string(elapsed_time)}"
            f" {elapsed_time / num_steps:.4f}s/step)"
        )

        ax1.set_title(plot_title)
        ax1.plot(episode_log["reward"], "g")
        # ax1.plot(
        #     lowess(
        #         episode_log["reward"],
        #         episode_log["episode num"],
        #         frac=0.1,
        #         return_sorted=False,
        #     ),
        #     "k",
        # )

        ax2.plot(step_log["reward"], "g")
        # ax2.plot(
        #     lowess(
        #         step_log["reward"],
        #         list(range(num_steps)),
        #         frac=0.1,
        #         return_sorted=False,
        #     ),
        #     "k",
        # )

        # l_step.set_ydata(step_log["step time"])
        # l_act.set_ydata(step_log["act time"])
        # l_train.set_ydata(step_log["train time"])
        # l_env.set_ydata(step_log["env time"])

        (l_step,) = ax3.plot(step_log["step time"], ".", color="k")
        (l_act,) = ax3.plot(step_log["act time"], ".", color="b")
        (l_train,) = ax3.plot(step_log["train time"], ".", color="r")
        (l_env,) = ax3.plot(step_log["env time"], ".", color="g")

        if first_plot:
            first_plot = False
            l_step.set_label("step")
            l_act.set_label("act")
            l_train.set_label("train")
            l_env.set_label("env")
            ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        return figure

    return figure, update_plot
