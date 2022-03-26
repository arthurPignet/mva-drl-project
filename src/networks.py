# @title Imports  { form-width: "30%" }

from acme import specs
import chex

import haiku as hk

import jax
import jax.numpy as jnp
import numpy as np
from typing import *


class ValueNetwork(hk.Module):
    def __init__(self, output_sizes: Sequence[int], name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._output_sizes = output_sizes

    def __call__(self, x: chex.Array) -> chex.Array:
        h = x

        for i, o in enumerate(self._output_sizes):
            h = hk.Linear(o)(h)
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            h = jax.nn.relu(h)
        return hk.Linear(1)(h)[..., 0]


class PolicyNetwork(hk.Module):
    def __init__(self, output_sizes: Sequence[int], action_spec: specs.BoundedArray,
                 name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._output_sizes = output_sizes
        self._action_spec = action_spec

    def __call__(self, x: chex.Array, ) -> Tuple[chex.Array, chex.Array]:
        action_shape = self._action_spec.shape
        action_dims = np.prod(action_shape)
        h = x
        for i, o in enumerate(self._output_sizes):
            h = hk.Linear(o)(h)
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
            h = jax.nn.relu(h)
        h = hk.Linear(2 * action_dims)(h)
        mu, pre_sigma = jnp.split(h, 2, axis=-1)
        sigma = jax.nn.softplus(pre_sigma)
        return hk.Reshape(action_shape)(.1 * mu), hk.Reshape(action_shape)(.1 * sigma)


class ValueNetworkDDPG(hk.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def __call__(self, s: chex.Array, a: chex.Array) -> chex.Array:
        h = s
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h)

        h = hk.Linear(400)(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.99)(h)
        h = jax.nn.relu(h)

        h = hk.Linear(300)(jnp.concatenate([h,a], axis=1))
        h = jax.nn.relu(h)

        return hk.Linear(1, hk.initializers.RandomUniform(-3e-3, 3e-3))(h)[..., 0]

class PolicyNetworkDDPG(hk.Module):
    def __init__(self, action_spec: specs.BoundedArray, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._action_spec = action_spec

    def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
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
        h = jnp.tanh(h)
        return hk.Reshape(action_shape)(h)
