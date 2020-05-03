import gym

import torch

# import numpy as np
import streamlit as st
import os
import pickle

from sac_openai_cuda import sac
from sac_openai_cuda_nets import MLPActorCritic
from plot_run_logs import plot_run_logs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def env_fn():
    return gym.make("BipedalWalker-v2")


latest_run = sorted(os.listdir("model_runs"))[-1]
st.write()


def run_selector(folder_path="."):
    dir_names = os.listdir("model_runs")
    dirs_by_dir_labels = {
        f'{dirname} ({len(os.listdir(f"model_runs/{dirname}"))} checkpoints)': dirname
        for dirname in dir_names
    }
    selected_label = st.selectbox("Select a run", sorted(list(dirs_by_dir_labels)))
    selected_dir = dirs_by_dir_labels[selected_label]
    return os.path.join(folder_path, selected_dir)


run_dirname = run_selector("model_runs")
st.write("You selected run `%s`" % run_dirname)


with open(run_dirname + "/log.pkl", "rb") as pickle_file:
    # print(pickle_file)
    log_info = pickle.load(pickle_file)
# print(list(log_info))
st.write(log_info["run params"])

episode_log = log_info["episode log"]
step_log = log_info["step log"]

fig, update_fn = plot_run_logs(episode_log, step_log)
update_fn(True)
st.pyplot(fig)
# st.write(np.un)
print()


def checkpoint_selector(folder_path="."):
    filenames = [f for f in sorted(os.listdir(folder_path)) if f[:5] == "model"]
    selected_filename = st.selectbox("Select a file", sorted(filenames))
    return os.path.join(folder_path, selected_filename)


checkpoint_filename = checkpoint_selector(run_dirname)
st.write("You selected model checkpoint `%s`" % checkpoint_filename)


def load_and_run_trained_net(checkpoint_filename, device):

    env = gym.make("BipedalWalker-v2")

    actor_net = MLPActorCritic(env.observation_space, env.action_space)

    actor_net.load_state_dict(torch.load(checkpoint_filename))
    actor_net.to(device)
    # print(actor_net)
    for i in range(10):
        print(i)
        state = env.reset()
        for j in range(200):
            # action = agent.act(state)
            env.render()
            action = actor_net.act(
                torch.as_tensor(state, dtype=torch.float32).to(device),
                # deterministic=True
            )
            # action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()


run_trained_model = st.button("run trained model (in new window)")
if run_trained_model:
    load_and_run_trained_net(checkpoint_filename, device)


def newAcFromOld(old_net):
    def constructor(*args, **kwargs):
        new_net = MLPActorCritic(*args, **kwargs)
        new_net.pi = old_net.pi
        new_net.q1 = old_net.q1
        new_net.q2 = old_net.q2
        new_net.device = old_net.device
        return new_net

    return constructor


train_more = st.button("keep training")
if train_more:
    env = gym.make("BipedalWalker-v2")
    actor_net = MLPActorCritic(env.observation_space, env.action_space)

    actor_net.load_state_dict(torch.load(checkpoint_filename))
    actor_net.to(device)

    st_more_training_plot = st.empty()

    actor_net2 = sac(
        env_fn,
        actor_critic=newAcFromOld(actor_net),  #########   <<----  WATCH OUT
        start_steps=0,
        steps_per_epoch=100,
        # max_ep_len=100,
        update_every=1,
        epochs=200,
        use_logger=False,
        alpha=0.2,
        lr=0.0005,
        # batch_size=256,
        # lr=7e-3,
        add_noise=True,
        device=device,
        st_plot_fn=st_more_training_plot.pyplot,
    )
# run_trained_net(actor_net)

# actor_net = MLPActorCritic(env_fn)
# actor_net, step_log, episode_log  = sac(
#     env_fn,
#     steps_per_epoch=0,
#     epochs=0,
#     use_logger=False,
#     # device=device
# )

# env = gym.make("BipedalWalker-v2")
# for i in range(10):
#     print(i)
#     state = env.reset()
#     while True:
#         # action = agent.act(state)
#         env.render()
#         action = actor_net.act(
#             torch.as_tensor(state, dtype=torch.float32), deterministic=True
#         )
#         # action = policy_net.get_action(state)
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         if done:
#             break

# env.close()


# N_runs = st.number_input()

# ac, step_log, episode_log = sac(
#     env_fn,
#     start_steps=20000,
#     steps_per_epoch=10000,
#     epochs=300,
#     use_logger=False,
#     update_every=50,
#     # alpha=.5,
#     batch_size=256,
#     lr=0.0001,
#     device=device,
#     st_plot=st_plot.pyplot,
# )

# plt.show()