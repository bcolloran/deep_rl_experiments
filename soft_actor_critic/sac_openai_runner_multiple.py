import gym

import torch

# import numpy as np
# import streamlit as st
# import os
# import pickle
import time

# import ray

import itertools
from collections import namedtuple

from sac_openai_cuda import sac, OUNoise

# from sac_openai_cuda_nets import MLPActorCritic
from plot_run_logs import plot_run_logs

# ray.init()

seeds = [1, 2, 3]
hidden_sizes = [(64, 64), (128, 64), (128, 128)]
lrs = [0.002, 0.001, 0.00005]
batch_sizes = [64, 128, 256]
alphas = [0.1, 0.2, 0.04]
add_noise = [True, False]

params = [seeds, hidden_sizes, lrs, batch_sizes, alphas, add_noise]

param_tup = namedtuple(
    "params", ["seed", "hidden_size", "lr", "batch_size", "alpha", "add_noise"]
)

param_tups = [param_tup(*p) for p in itertools.product(*params)]
param_tups = [tuple(p) for p in itertools.product(*params)]


# for element in itertools.product(*params):
#     param_tups.append(param_tup(*element))


# @ray.remote(num_cpus=8)
# @ray.remote
def train_model(param_tup):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)

    #         "BipedalWalker-v2",
    #         "Pendulum-v0",
    #         "BipedalWalkerHardcore-v2",
    #         "MountainCarContinuous-v0",
    #         "LunarLanderContinuous-v2",
    selected_env = "BipedalWalker-v2"

    steps_per_epoch = 10000

    epochs = 20

    start_steps = 10000
    print(f"running 'train_model' with params {param_tup}")
    (seed, hidden_sizes, lr, batch_size, alpha, add_noise) = param_tup

    def env_fn():
        return gym.make(selected_env)

    sac(
        env_fn,
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        alpha=alpha,
        add_noise=add_noise,
        env_name=selected_env,
        start_steps=start_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        ac_kwargs={"hidden_sizes": hidden_sizes},
        device=device,
    )


# train_model.remote(param_tups[0])
# train_model(param_tups[0])

# for i, pt in enumerate(param_tups):
#     print(f"param set {i} of {len(param_tups)};\n    {pt}")
#     ac, step_log, episode_log = train_model.remote(pt)
# train_model(pt)

# futures = [train_model.remote(pt) for pt in param_tups[:20]]

# x = 23
# print(ac, step_log, episode_log)

# @ray.remote(num_cpus=4)
# def f(i):
#     time.sleep(1)
#     # raise ValueError
#     return i + x


# futures = [train_model.remote(i) for i in range(20)]
# print(ray.get(futures))
# ray.timeline(filename="tmp/timeline.json")

if __name__ == "__main__":
    from multiprocessing import Pool

    # info('main line')
    # p = Process(target=f, args=('bob',))
    # p.start()
    # p.join()
    with Pool(7) as p:
        p.map(train_model, param_tups)
