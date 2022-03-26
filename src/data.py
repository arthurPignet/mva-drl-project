from acme import types
import chex
import optax
from typing import Mapping


@chex.dataclass
class Trajectory:
    observations: types.NestedArray  # [T, B, ...]
    actions: types.NestedArray  # [T, B, ...]
    rewards: chex.ArrayNumpy  # [T, B]
    dones: chex.ArrayNumpy  # [T, B]
    discounts: chex.ArrayNumpy  # [T, B]c


@chex.dataclass
class LearnerState:
    params: chex.Array
    state: chex.Array
    opt_state: optax.OptState

LogsDict = Mapping[str, chex.Array]
