# @title Imports  { form-width: "30%" }

from acme import specs
import chex

import haiku as hk

import jax
import jax.numpy as jnp
import numpy as np
from typing import *


class ValueNetworkDDPG(hk.Module):
    """
    DDPG value network, following the original paper
    DDPG Q function network, following the original paper
    Linear(300) -> Linear(400) -> Linear(1), with BatchNorm and ReLU
    Actions are only taken into account at the second hidden layer.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def __call__(self, s: chex.Array, a: chex.Array, is_training: bool) -> chex.Array:
        """
        :param s: state
        :param a: action
        :param is_training: True if training mode, for BatchNorm
        :return: Q(s,a)
        """

        h = s
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h, is_training)

        h = hk.Linear(400)(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h, is_training)
        h = jax.nn.relu(h)

        h = hk.Linear(300)(jnp.concatenate([h,a], axis=1))
        h = jax.nn.relu(h)

        return hk.Linear(1, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)[..., 0]

class PolicyNetworkDDPG(hk.Module):
    """
    DDPG policy network, following the original paper
    Linear(300) -> Linear(400) -> Linear(action shape) -> tanh, with BatchNorm and ReLU
    """

    def __init__(self, action_spec: specs.BoundedArray, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._action_spec = action_spec

    def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
        """
        :param x: state
        :param is_training: True if training mode, for BatchNorm
        :return: policy(x)
        """
        
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)
        h = x
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h, is_training)

        h = hk.Linear(400)(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h, is_training)
        h = jax.nn.relu(h)

        h = hk.Linear(300)(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h, is_training)
        h = jax.nn.relu(h)

        h = hk.Linear(action_dims, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)
        h = jnp.tanh(h) * self._action_spec.maximum # tanh is bounded between -1 and 1.
        return hk.Reshape(action_shape)(h) 
