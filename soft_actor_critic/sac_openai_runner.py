import gym

import torch


from sac_openai_cuda import sac

import streamlit as st


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def env_fn():
    return gym.make("BipedalWalker-v2")


st_plot = st.empty()

ac, step_log, episode_log = sac(
    env_fn,
    start_steps=20000,
    steps_per_epoch=10000,
    epochs=300,
    # use_logger=False,
    update_every=50,
    # alpha=.5,
    batch_size=256,
    lr=0.0001,
    device=device,
    # st_plot=st_plot.pyplot,
)

# plt.show()
