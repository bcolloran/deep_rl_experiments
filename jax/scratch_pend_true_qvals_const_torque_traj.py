# %%
import numpy as np
from numpy.random import randn, rand

# import time
# import jax
from jax import grad, value_and_grad, random, jit, jacfwd
import jax.numpy as jnp

# from jax.ops import index, index_add, index_update

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import OrderedDict as odict


from IPython import get_ipython

from importlib import reload

import jax_nn_utils as jnn
from damped_spring_noise import dampedSpringNoise
import pendulum_utils as PU
import pendulum_plotting as PP
import simple_logger as SL

reload(PU)
reload(PP)
reload(SL)
reload(jnn)


get_ipython().run_line_magic("matplotlib", "inline")

params = PU.default_pendulum_params


T = 300
plotter2 = PP.PendulumValuePlotter2(n_grid=100, jupyter=True)
plotter2.plotX.shape
im_plots = []
for torque in np.linspace(-2, 2, 5):
    discounted_cost = np.zeros(plotter2.plotX.shape[1])
    for i in range(plotter2.plotX.shape[1]):
        theta0 = plotter2.plotX[0, i]
        thetadot0 = plotter2.plotX[1, i]
        traj = PU.pendulumTraj_scan(theta0, thetadot0, torque * np.ones(T))
        discounted_cost[i] = np.sum(traj[:, 2] * discount ** np.arange(T))
    im_plots.append((f"torque={torque}", discounted_cost))

plotter2.update_plot(
    im_plots,
    title=f"true discounted cost of traj from point, const torque\nT={T}, disc={discount}",
)
