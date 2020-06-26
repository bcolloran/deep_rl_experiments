from typing import Callable, NewType, Any, Tuple, TypeVar, Union, Dict, Hashable

# from numpy import ndarray
import jax.numpy as jnp

# from noise_procs import Noise


class jax_array:
    def __getitem__(self, indices) -> Any:
        ...


class np_array:
    def __getitem__(self, indices) -> Any:
        ...


Tensor = Union[np_array, jax_array]

JaxArray = NewType("JaxArray", jax_array)
# SAR and trajectories
State = Tensor
Action = Tensor
Reward = Tensor
SarTup = Tuple[State, Action, Reward]
SrTup = Tuple[State, Reward]
SarTrajTup = Tuple[State, Action, Reward]


NNParams = Any
NNParamsFn = Callable[[NNParams, Tensor], Any]

# FUNCTIONS
# DynamicsFn = Callable[[State, Action], Tuple[State, Reward]]
# PolicyFn = Callable[[State], Action]
# # NoisePolicyFn = Callable[[State, Noise], Action]
# # NoisePolicyNetFn = Callable[[NNParams, State, Noise], Action]

# SCANABLES
StateScanTup = Tuple[State, SarTup]
StateScannableFn = Callable[[State, Any], StateScanTup]

