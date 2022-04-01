import abc
import functools

import numpy as np
import jax
import rlax
import jax.numpy as jnp
import optax
import chex
from .data import Trajectory, LearnerState, LogsDict, Transition
import tree
from typing import Sequence, Mapping, Tuple
from acme import types, specs
from functools import partial
import haiku as hk

from .networks import PolicyNetwork, ValueNetwork, PolicyNetworkDDPG, ValueNetworkDDPG


# A very simple agent API, with just enough to interact with the environment
# and to update its potential parameters.
class Agent(abc.ABC):
    @abc.abstractmethod
    def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
        """One step of learning on a trajectory.

        The mapping returned can contain various logs.
        """
        pass

    @abc.abstractmethod
    def batched_actor_step(self, observations: types.NestedArray) -> types.NestedArray:
        """Returns actions in response to observations.

        Observations are assumed to be batched, i.e. they are typically arrays, or
        nests (think nested dictionaries) of arrays with shape (B, F_1, F_2, ...)
        where B is the batch size, and F_1, F_2, ... are feature dimensions.
        """
        pass


class RandomAgent(Agent):
    def __init__(self, environment_spec: specs.EnvironmentSpec) -> None:
        self._action_spec = environment_spec.actions

    def batched_actor_step(self, observation: types.NestedArray) -> types.NestedArray:
        batch_size = tree.flatten(observation)[0].shape[0]
        return np.random.randn(batch_size, *self._action_spec.shape)

    def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
        # Skip it
        return dict()






class DDPGAgent(Agent):
    '''
    Implement a Deep Deterministic Policy Gradient agent, as described in the paper 'Continious control with deep reinforcement learning'
    
    '''
    def __init__(self, seed: int, actor_learning_rate: float, critic_learning_rate: float, gamma: float, tau: float, environment_spec: specs.EnvironmentSpec) -> None:
        self._rng = jax.random.PRNGKey(seed=seed)
        # hk.transform_with_state because of BatchNorm
        self._init_critic_loss, apply_critic_loss = hk.without_apply_rng(hk.transform_with_state(self._critic_loss_function))
        self._init_actor_loss, apply_actor_loss = hk.without_apply_rng(hk.transform_with_state(self._actor_loss_function))       
       # modify structure of output for jax.value_and_grad
        def _critic_apply_loss_2( *args, **kwargs):
            (loss, aux), state = apply_critic_loss(*args,**kwargs)
            return loss, (aux, state)
        def _actor_apply_loss_2( *args, **kwargs):
            (loss, aux), state = apply_actor_loss(*args,**kwargs)
            return loss, (aux, state)
        self._grad_actor = jax.value_and_grad(_actor_apply_loss_2, has_aux=True)
        self._grad_critic = jax.value_and_grad(_critic_apply_loss_2, has_aux=True)

        _, self._apply_policy = hk.without_apply_rng(hk.transform_with_state(self._hk_apply_policy))
        _, self._apply_value = hk.without_apply_rng(hk.transform_with_state(self._hk_apply_value))

        #self._entropy_loss_coef = entropy_loss_coef  don't need that as we are not regulazing
        self._gamma = gamma
        self._tau = tau
        self._environment_spec = environment_spec

        self._critic_optimizer = optax.adam(learning_rate=critic_learning_rate)
        self._actor_optimizer = optax.adam(learning_rate=actor_learning_rate)

        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)

        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self.init_fn(init_rng, self._generate_dummy_transition())

    def _generate_dummy_transition(self) -> Trajectory:
        observation = self._environment_spec.observations.generate_value()
        action = self._environment_spec.actions.generate_value()

        return Transition(
            obs_tm1= observation[None],
            action_tm1= action[None],
            reward_t= jnp.zeros(1)[None],
            discount_t=jnp.zeros(1)[None],
            obs_t=observation[None],
            done=jnp.zeros(1)[None]
        )

    def _init_fn(self, rng: chex.PRNGKey, transition: Transition) -> LearnerState:
        critic_params, critic_state = self._init_critic_loss(rng, transition, True)
        actor_params, actor_state = self._init_actor_loss(rng, transition, True)

        # get only critic, resp. actor params. 
        critic_params = {k:critic_params[k] for k in critic_params.keys() if  k.startswith("value/")}
        actor_params = {k:actor_params[k] for k in actor_params.keys() if  k.startswith("policy/")}
        # copy the params for the target networks. 
        target_critic_params = {k.replace("value/", "value_target/"):critic_params[k] for k in critic_params.keys()}
        target_actor_params = {k.replace("policy/", "policy_target/"):actor_params[k] for k in actor_params.keys()}

        opt_critic_state = self._critic_optimizer.init(critic_params)
        opt_actor_state = self._actor_optimizer.init(actor_params)
        actor_state.update(critic_state)

        actor_params.update(critic_params)
        actor_params.update(target_actor_params)
        actor_params.update(target_critic_params)

        return LearnerState(params=actor_params, state=actor_state, opt_critic_state=opt_critic_state, opt_actor_state=opt_actor_state)

    def slow_copy(self, params):
        tau = self._tau
        for k, v in params.items():
            if k.startswith(("value/", "policy/")):
                k_t = k.replace("value/", "value_target/").replace("policy/", "policy_target/")
                params[k_t] = jax.tree_util.tree_map(lambda x, y: tau * x + (1 - tau) * y, v, params[k_t])
        return params

    def _update_fn(self, learner_state: LearnerState, transition: Transition) -> Tuple[LearnerState, LogsDict]:
        # critic update
        #   Compute the critic loss and the gradients for ALL the parameters value, policy and targets
        (critic_loss, (aux_critic, critic_state)), critic_grads = self._grad_critic(learner_state.params, learner_state.state,  transition)

        # Select the only params and gradients we are interested in for this part, the critics
        critic_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("value/")}
        critic_grads_only_value = {k:critic_grads[k] for k in critic_grads.keys() if k.startswith("value/")}

        # Perform one optimization step
        critic_udpates, new_opt_critic_state = self._critic_optimizer.update(critic_grads_only_value, learner_state.opt_critic_state, critic_params)
        new_critic_params = optax.apply_updates(critic_params, critic_udpates)

        # Update the critic
        learner_state.params.update(new_critic_params)
        learner_state.state.update(critic_state)  # As the value network is used with is_training=False, we need to update the state. 

        # actor update
        #   Compute the actor loss and the gradients for ALL the parameters value, policy and targets
        (actor_loss, (aux_actor, actor_state)), actor_grads = self._grad_actor(learner_state.params, learner_state.state,  transition)

        # Select the only params and gradients we are interested in for this part, the actors
        actor_params = {k:learner_state.params[k] for k in learner_state.params.keys() if  k.startswith("policy/")}
        actor_grads_only_policy = {k:actor_grads[k] for k in actor_grads.keys() if k.startswith("policy/")}

        # Perform one optimization step
        actor_updates, new_opt_actor_state = self._actor_optimizer.update(actor_grads_only_policy, learner_state.opt_actor_state, actor_params)
        new_actor_params = optax.apply_updates(actor_params, actor_updates)

        # Update the critic
        learner_state.params.update(new_actor_params)
        learner_state.state.update(actor_state)

        # Update the target networks using slow copy
        new_params = self.slow_copy(learner_state.params) 
        state = self.slow_copy(learner_state.state)

        # aggregate the logs
        aux_critic.update(aux_actor)

        return LearnerState(params=new_params, state=state, opt_critic_state=new_opt_critic_state, opt_actor_state=new_opt_actor_state), aux_critic

    def learner_step(self, transition: Transition) -> Mapping[str, chex.ArrayNumpy]:
        self._learner_state, logs = self.update_fn(self._learner_state, transition)
        return logs

    def _batched_actor_step(self, learner_state: LearnerState, rng: chex.PRNGKey, observations: types.NestedArray,
                            ) -> types.NestedArray:
        actions, state = self.apply_policy(learner_state.params, learner_state.state, observations)
        return actions #learner state does not need to be propagated as it's an off-policy algorithm

    def batched_actor_step(self, observations: types.NestedArray) -> types.NestedArray:
        """Returns actions in response to observations."""
        return self._batched_actor_step(self._learner_state, self._rng, observations)


    def _hk_apply_value(self, observations: types.NestedArray, actions:types.NestedArray) -> chex.Array:
        return ValueNetworkDDPG(name='value')(observations, actions) #2nd arg will be actions

    def _hk_apply_policy(self, observations: types.NestedArray) -> chex.Array:
        return PolicyNetworkDDPG(self._environment_spec.actions, name='policy')(observations, False)

    def _actor_loss_function(self, transition: Transition, is_init=False) -> Tuple[chex.Array, LogsDict]:
        # actor loss
        action_ = PolicyNetworkDDPG(self._environment_spec.actions, name='policy')(transition.obs_tm1, True)
        value_network = ValueNetworkDDPG(name='value',)
        values = value_network(transition.obs_tm1, action_, is_init) #2nd arg will be actions
        actor_loss = -jnp.mean(values)

        logs = dict(policy_loss=actor_loss)
        return actor_loss, logs


    def _critic_loss_function(self, transition: Transition, is_init=False) -> Tuple[chex.Array, LogsDict]:

        #critic loss
        q_value = ValueNetworkDDPG(name='value')(transition.obs_tm1, transition.action_tm1, True)
        next_action = PolicyNetworkDDPG(self._environment_spec.actions, name='policy_target')(transition.obs_t, True)
        bootstrapped_q = ValueNetworkDDPG(name='value_target')(transition.obs_t, next_action, is_init)
        q_target = jax.lax.stop_gradient(transition.reward_t + self._gamma *  (1-transition.done[..., None])  * bootstrapped_q)
       
        value_loss = jnp.mean(.5 * jnp.square(q_target - q_value))

        logs = dict(actions_mean=transition.action_tm1.mean(),
                    value_loss=value_loss,
                    value_mean=q_value.mean(),
                    value_target_mean=q_target.mean(),
                    mean_reward=transition.reward_t.mean(),
                    obs=transition.obs_tm1.mean(axis=(0, 1)),
                    done = transition.done.mean(),
                    bootstrapped_q = bootstrapped_q.mean()
                    )
        return value_loss, logs


class A2CAgent(Agent):
    def __init__(self, seed: int, learning_rate: float, gamma: float, value_output_sizes: Sequence[int],
                 policy_output_sizes: Sequence[int], environment_spec: specs.EnvironmentSpec,
                 entropy_loss_coef: float) -> None:
        self._rng = jax.random.PRNGKey(seed=seed)
        self._init_loss, apply_loss = hk.without_apply_rng(hk.transform(self._loss_function))
        self._grad = jax.value_and_grad(apply_loss, has_aux=True)
        _, self._apply_policy = hk.without_apply_rng(hk.transform(self._hk_apply_policy))
        _, self._apply_value = hk.without_apply_rng(hk.transform(self._hk_apply_value))

        self._value_output_sizes = value_output_sizes
        self._policy_output_sizes = policy_output_sizes
        self._entropy_loss_coef = entropy_loss_coef
        self._gamma = gamma
        self._environment_spec = environment_spec

        self._optimizer = optax.adam(learning_rate=learning_rate)
        self.init_fn = jax.jit(self._init_fn)
        self.update_fn = jax.jit(self._update_fn)
        self.apply_policy = jax.jit(self._apply_policy)
        self.apply_value = jax.jit(self._apply_value)

        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self._init_fn(init_rng, self._generate_dummy_trajectory())

    def _init_fn(self, rng: chex.PRNGKey, trajectory: Trajectory) -> LearnerState:
        params = self._init_loss(rng, trajectory)
        opt_state = self._optimizer.init(params)
        return LearnerState(params=params, opt_state=opt_state)

    def _update_fn(self, learner_state: LearnerState, trajectory: Trajectory) -> Tuple[LearnerState, LogsDict]:
        (loss, aux), grads = self._grad(learner_state.params, trajectory)
        udpates, new_opt_state = self._optimizer.update(grads, learner_state.opt_state, learner_state.params)
        new_params = optax.apply_updates(learner_state.params, udpates)
        return LearnerState(params=new_params, opt_state=new_opt_state), aux

    def learner_step(self, trajectory: Trajectory) -> Mapping[str, chex.ArrayNumpy]:
        self._learner_state, logs = self.update_fn(self._learner_state, trajectory)
        return logs

    def _batched_actor_step(self, learner_state: LearnerState, rng: chex.PRNGKey, observations: types.NestedArray,
                            for_eval: bool = False) -> types.NestedArray:
        mu, sigma = self.apply_policy(learner_state.params, observations)

        if for_eval:
            actions = rlax.gaussian_diagonal().sample(rng, mu, 0. * sigma)
        else:
            actions = rlax.gaussian_diagonal().sample(rng, mu, sigma)
        return actions

    def _generate_dummy_trajectory(self) -> Trajectory:
        observation = self._environment_spec.observations.generate_value()
        action = self._environment_spec.actions.generate_value()

        def _add_dim(x: types.NestedArray, dim_size: int) -> types.NestedArray:
            return jax.tree_map(lambda x: jnp.repeat(x[None], axis=0, repeats=dim_size), x)

        return Trajectory(
            observations=_add_dim(_add_dim(observation, 2), 2),
            actions=_add_dim(_add_dim(action, 2), 2),
            rewards=jnp.zeros((2, 2)),
            dones=jnp.zeros((2, 2)),
            discounts=jnp.zeros((2, 2)),

        )

    def batched_actor_step(self, observations: types.NestedArray, for_eval: bool = False) -> types.NestedArray:
        """Returns actions in response to observations."""
        return self._batched_actor_step(self._learner_state, self._rng, observations, for_eval)

    def _hk_apply_value(self, observations: types.NestedArray) -> Tuple[chex.Array, chex.Array]:
        return ValueNetwork(self._value_output_sizes)(observations)

    def _hk_apply_policy(self, observations: types.NestedArray) -> Tuple[chex.Array, chex.Array]:
        return PolicyNetwork(self._policy_output_sizes, self._environment_spec.actions)(observations)

    def _loss_function(self, trajectory: Trajectory) -> Tuple[chex.Array, LogsDict]:
        """Implement A2C's loss function.

        This function will be passed in to a hk.transform, the resulting
        transformed tuple (init_fn, apply_fn) should be such that:
        - init_fn initializes the parameters of both the actor and critic networks
          given a random number generator and a trajectory.
        - apply_fn should:
          1. Return the correct sum of critic and actor losses, given as input the
             parameters of the actor and critic networks.
          2. More importantly returns the correct gradients when passed through
             jax.grad, i.e. the gradients defined above. To that end, you will
             have to cleverly use the jax.lax.stop_gradient facility, which
             prevents the gradient from flowing in certain subparts of the
             computational graph.

        Hint:
        As a reminder, trajectories are provided such that the full sequence that
        we get is
        o_0 a_0 r_0 d_0, ...., o_T, a_T, r_T, d_T
        """
        # inputs are assumed to be provided such that the full sequence that we get is
        # o_0 a_0 r_0 d_0, ...., o_T, a_T, r_T, d_T
        print(trajectory.observations.shape)
        mu, sigma = hk.BatchApply(PolicyNetwork(self._policy_output_sizes, self._environment_spec.actions))(
            trajectory.observations)
        values = hk.BatchApply(ValueNetwork(self._value_output_sizes))(trajectory.observations)

        batched_return_fn = jax.vmap(
            functools.partial(rlax.lambda_returns, stop_target_gradients=True),
            in_axes=1,
            out_axes=1)
        value_targets = batched_return_fn(
            trajectory.rewards[1:],
            (self._gamma * trajectory.discounts * (1. - trajectory.dones))[:-1],
            values[1:],
        )
        value_loss = jnp.mean(.5 * jnp.square(values[:-1] - value_targets))

        sg_advantages = jax.lax.stop_gradient(value_targets - values[:-1])
        action_log_probs = rlax.gaussian_diagonal().logprob(trajectory.actions, mu, sigma)

        entropies = rlax.gaussian_diagonal().entropy(mu, sigma)
        entropy_loss = -jnp.mean(entropies)
        policy_loss = -jnp.mean(sg_advantages * action_log_probs[:-1])
        policy_loss = policy_loss + self._entropy_loss_coef * entropy_loss

        logs = dict(actions_mean=trajectory.actions.mean(),
                    value_loss=value_loss,
                    policy_loss=policy_loss,
                    entropy_loss=entropy_loss,
                    value_mean=values.mean(),
                    value_target_mean=value_targets.mean(),
                    mean_reward=trajectory.rewards.mean(),
                    obs=trajectory.observations.mean(axis=(0, 1)),
                    )
        return value_loss + policy_loss, logs