#%%
# import numpy
import time

# from numpy import random

import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as np
from jax.ops import index, index_add, index_update

import matplotlib.pyplot as plt


from importlib import reload
import pendulum_utils
import trees

reload(pendulum_utils)
reload(trees)
# %%

# %%
