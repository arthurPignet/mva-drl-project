# @title Imports  { form-width: "30%" }
import dm_env

import multiprocessing as mp
import multiprocessing.connection

from typing import *
from .data import Trajectory

import chex
import numpy as np
import tree


def _validate_timestep(ts: dm_env.TimeStep) -> dm_env.TimeStep:
    """Some timesteps may come with rewards or discounts set to None, we
  replace such values by 0. (resp 1.)."""
    if ts.reward is None or ts.discount is None:
        ts = ts._replace(reward=0., discount=1.)
    return ts


def actor_target(env_factory: Callable[[], dm_env.Environment], idx: int, controller_handle: mp.connection.Connection,
                 max_steps: Optional[int] = None) -> None:
    environment = env_factory(seed=idx)
    ts = environment.reset()

    ts = _validate_timestep(ts)

    controller_handle.send(ts)

    steps = 0
    while True:
        if max_steps is not None and steps >= max_steps:
            break
        action = controller_handle.recv()
        ts = environment.step(action)
        if ts.last():
            reset_ts = environment.reset()
            ts = ts._replace(observation=reset_ts.observation)

        ts = _validate_timestep(ts)

        controller_handle.send(ts)
        steps += 1


def sequence_of_timesteps_with_action_to_trajectory(
        timesteps_with_actions: Sequence[Tuple[dm_env.TimeStep, chex.Array]],
) -> Trajectory:
    stacked_timesteps, actions = tree.map_structure(lambda *x: np.stack(x, axis=0), *timesteps_with_actions)
    observations = stacked_timesteps.observation[:-1].astype(np.float32)
    actions = actions[:-1].astype(np.float32)
    rewards = stacked_timesteps.reward[1:].astype(np.float32)
    dones = (stacked_timesteps.step_type[1:] == dm_env.StepType.LAST).astype(np.float32)
    discounts = stacked_timesteps.discount[1:].astype(np.float32)
    return Trajectory(observations=observations,
                      actions=actions,
                      rewards=rewards,
                      dones=dones,
                      discounts=discounts)


# Evaluation facilities
def returns(rewards: chex.ArrayNumpy, dones: chex.ArrayNumpy) -> chex.ArrayNumpy:
    r = np.zeros_like(rewards[0], dtype=np.float32)
    for r_t, d_t in zip(rewards[::-1], dones[::-1]):
        r = r_t + (1. - d_t) * r
    return r
