from acme import types
import chex
import optax
from typing import Mapping
import collections


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
    state: chex.Array # state for the batchNorm and other stuff that haiku need (cf transform_with_state)
    opt_critic_state: optax.OptState
    opt_actor_state: optax.OptState


LogsDict = Mapping[str, chex.Array]

Transition = collections.namedtuple("Transition", field_names=["obs_tm1", "action_tm1", "reward_t", "discount_t", "obs_t", "done"])
