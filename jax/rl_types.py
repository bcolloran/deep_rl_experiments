from typing import Callable, NewType, Any, Tuple, TypeVar, Union, Dict, Hashable, Sized
from typing_extensions import Protocol

# from numpy import ndarray
import jax.numpy as jnp


class Tensor(Sized):
    def __getitem__(self, indices) -> Any:
        ...

    def __mul__(self, other: Any) -> "Tensor":
        ...

    def __pow__(self, other: Any) -> "Tensor":
        ...

    def flatten(self) -> "Tensor":
        ...


# SAR and trajectories
State = Tensor
Action = Tensor
Reward = Tensor
SarTup = Tuple[State, Action, Reward]
SrTup = Tuple[State, Reward]
SarTrajTup = Tuple[State, Action, Reward]


NNParams = Any
# NNParamsFn = Callable[[NNParams, Tensor, ...], Any]
class NNParamsFn(Protocol):
    def __call__(self, nn_params: NNParams, *args: Tensor) -> Tuple[Tensor, ...]:
        ...


# FUNCTIONS
# DynamicsFn = Callable[[State, Action], Tuple[State, Reward]]
# PolicyFn = Callable[[State], Action]
# # NoisePolicyFn = Callable[[State, Noise], Action]
# # NoisePolicyNetFn = Callable[[NNParams, State, Noise], Action]

# SCANABLES
StateScanTup = Tuple[State, SarTup]
StateScannableFn = Callable[[State, Any], StateScanTup]

