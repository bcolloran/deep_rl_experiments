def 
f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
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

    steps_to_average = 10000
    episodes_to_average = 100

    def update_plot(first_plot):
        t = len(step_log["reward"])

        elapsed_time = np.round(time.time() - start_time)

        plot_title = (
            f"episode {episode_num};  step {t}"
            f"\nRecent avg step score: {np.mean(step_log['reward'][-steps_to_average:]):.2f}"
            f"\ntime: {elapsed_time_string(elapsed_time)}"
            f" {(time.time() - start_time) / t:.4f}s/step)"
        )

        ax1.set_title(plot_title)
        ax1.plot(episode_log["reward"], "g")
        ax1.plot(running_mean(episode_log["reward"], -episodes_to_average), "k")

        ax2.plot(step_log["reward"], "g")
        ax2.plot(running_mean(step_log["reward"], -steps_to_average), "k")

        # l_step.set_ydata(step_log["step time"])
        # l_act.set_ydata(step_log["act time"])
        # l_train.set_ydata(step_log["train time"])
        # l_env.set_ydata(step_log["env time"])

        (l_step,) = ax3.plot(step_log["step time"], color="k")
        (l_act,) = ax3.plot(step_log["act time"], color="b")
        (l_train,) = ax3.plot(step_log["train time"], color="r")
        (l_env,) = ax3.plot(step_log["env time"], color="g")

        if first_plot:
            first_plot = False
            l_step.set_label("step")
            l_act.set_label("act")
            l_train.set_label("train")
            l_env.set_label("env")
            ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.show(block=False)

        # display.clear_output(wait=True)
        # display.display(f)
        return first_plot