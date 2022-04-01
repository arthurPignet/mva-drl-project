import chex
import matplotlib.pyplot as plt
import numpy as np
import jax
from typing import Mapping

# Pretty printing facility for logs.
def pretty_print(logs: Mapping[str, chex.Array]) -> str:
    def _process_array(x: chex.Array) -> str:
        x = jax.device_get(x)
        if x.ndim == 0:
            return f'{x:.2f}'
        return ', '.join([f'{v:.2f}' for v in x])

    logs = jax.tree_map(lambda x: _process_array(x), logs)
    return '\t|'.join([f'{k}: {v}' for k, v in logs.items()])


# title Visualization function for pendulum
def plot_policy_on_pendulum(agent, num_bins: int, num_ticks: int = 6) -> None:
    theta = np.linspace(-np.pi, np.pi, num=num_bins)
    dtheta = np.linspace(-10., 10., num=num_bins)
    thetas, dthetas = np.meshgrid(theta, dtheta)
    xs, ys = np.cos(thetas), np.sin(thetas)
    obs = np.stack([xs, ys, dthetas], axis=-1)
    actions, _ = agent.apply_policy(agent._learner_state.params, agent._learner_state.state, np.reshape(obs, (-1, 3)))
    values, _ = agent.apply_value(agent._learner_state.params, agent._learner_state.state, np.reshape(obs, (-1, 3)), actions)
    num_divided_ticks = num_bins // num_ticks
    plt.subplot(121)
    plt.xticks(range(num_bins)[::num_divided_ticks], [f'{s:.1f}' for s in theta][::num_divided_ticks])
    plt.yticks(range(num_bins)[::num_divided_ticks], [f'{s:.1f}' for s in dtheta][::num_divided_ticks])
    plt.title('Policy')
    plt.xlabel('theta')
    plt.ylabel('dtheta_dt')
    plt.imshow(np.reshape(actions[..., 0], (num_bins, num_bins)))
    plt.subplot(122)
    plt.xticks(range(num_bins)[::num_divided_ticks], [f'{s:.1f}' for s in theta][::num_divided_ticks])
    plt.yticks(range(num_bins)[::num_divided_ticks], [f'{s:.1f}' for s in dtheta][::num_divided_ticks])
    plt.title('Value')
    plt.xlabel('theta')
    plt.imshow(np.reshape(values, (num_bins, num_bins)))
    plt.show()
