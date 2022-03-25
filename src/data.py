from acme import types
import chex
import optax


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
    opt_state: optax.OptState
