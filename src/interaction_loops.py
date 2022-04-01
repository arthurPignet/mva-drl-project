from copy import deepcopy
import dm_env

import multiprocessing as mp
import numpy as np
import jax

import tree
from typing import *

from .utils import actor_target, returns, my_process
from .plot_utils import pretty_print
from .agents import Agent, DDPGAgent
from .replay_buffer import BatchedReplayBuffer


def simple_interaction_loop(agent: Agent, environment: dm_env.Environment, max_num_steps: int = 5000) -> None:
    ts = environment.reset()
    for _ in range(max_num_steps):
        if ts.last():
            break
        print(ts.reward)
        batched_observation = tree.map_structure(lambda x: x[None], ts.observation)
        action = agent.batched_actor_step(batched_observation)[0]  # batch size = 1

        ts = environment.step(action)


def simple_parallel_interaction_loop(agent: Agent, env_factory: Callable[[], dm_env.Environment], total_num_steps: int,
                                     num_actors: int = 2) -> None:
    """This function should do the converse of what actor_target is doing, i.e.:

    - Spawn the actors and create pipes for the actors to communicate with the
      main process (this part of the code is provided).
    - For a given number of steps, receive the timesteps from the actors, stack
      them, and compute corresponding actions.
    - Once the actions are computed, send them to the actors through the pipes.
    """
    actor2learner_pipes, learner2actors_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
    max_steps = total_num_steps // num_actors if total_num_steps is not None else None

    actors = []
    for i, pipe in enumerate(actor2learner_pipes):
        process = mp.Process(target=actor_target, args=(env_factory, i, pipe, max_steps))
        process.start()
        actors.append(process)

    #####
    for _ in range(max_steps):
        all_ts_from_actors = [pipe.recv() for pipe in actor2learner_pipes]

        ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *all_ts_from_actors)
        actions = agent.batched_actor_step(ts.observation)

        for i, pipe in enumerate(learner2actors_pipes):
            pipe.send(actions[i])
    ####

    for actor in actors:
        actor.join()


def ddpg_parallel_interaction_loop(agent: DDPGAgent, env_factory: Callable[[], dm_env.Environment], max_learner_steps: int, replay: BatchedReplayBuffer, batch_size: int = 32,rewards_scaler=0.1, num_actors: int = 2, seed = 0):
  """
    Creates actors and run them in parallel, computes the actions and adds transitions to a replay buffer for training.
    :param agent: DDGP agent
    :param env_factory: creates the environment from a seed
    :param max_learner_steps: number of learning steps
    :param replay: replay buffer
    :param batch_size: size of the batch
    :param rewards_scaler: scales the rewards for stability
    :param num_actors: number of actors that are run in parallel
    :param seed: seed of environment
    :return: data for plotting purposes, replay buffer
    """
  plots = dict(iterations = [],
               actions_mean=[],
               value_loss=[],
               policy_loss=[],
               value_mean=[],
               value_target_mean=[],
               mean_reward=[],
               obs=[],
              bootstrapped_q=[],
              done=[]
               )
  rng_key = jax.random.PRNGKey(seed=seed)
  parent_pipes, children_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
  actors = [mp.Process(target=actor_target, args=(env_factory, np.random.randint(100), pipe, max_learner_steps)) for i, pipe in enumerate(children_pipes)]
  for actor in actors:
    actor.start()

  compt = 0
  for learner_step in range(max_learner_steps):
    ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *[pipe.recv() for pipe in parent_pipes])
    if learner_step>0:
      for i in range (len(ts.observation)):
        replay.add(timestep.observation[i],actions[i],rewards_scaler*ts.reward[i],ts.discount[i],ts.observation[i],(ts.step_type[i] == dm_env.StepType.LAST)*1.)
      if len(replay._memory) > batch_size:
        transitions = replay.sample_batch(min(batch_size,len(replay._memory)-1))
        logs = agent.learner_step(transitions)
        compt +=1
        if compt % 10 == 0:
          print(f'iteration nb{learner_step}')
          print(pretty_print(logs))
          for name in logs.keys():
            plots[name].append(logs[name])
          plots['iterations'].append(learner_step)

    actions = agent.batched_actor_step(ts.observation)
    rng_key, init_rng = jax.random.split(rng_key)
    #noisy_actions = actions + jax.random.normal(key=init_rng, shape=actions.shape)*0.2
    actions = actions + my_process(x0=0,paths=1)(np.linspace(0,1,actions.shape[0]))
    for i, pipe in enumerate(parent_pipes):
      pipe.send(actions[i])
    timestep = deepcopy(ts)

  for actor in actors:
    actor.join()
  return plots, replay

def ddpg_evaluation_parallel_interaction_loop(agent: DDPGAgent,
                                              env_factory: Callable[[],
                                              dm_env.Environment],
                                              sequence_length: int,
                                              num_actors: int = 2) -> float:
  """
    Creates actors and run them in parallel for evaluation of the agent.
    :param agent: DDPG agent
    :param env_factory: creates the environment from a seed
    :param sequence_length: number of steps in evaluation
    :param num_actors: number of actors that are run in parallel
    :return: average return
  """
  parent_pipes, children_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
  actors = [mp.Process(target=actor_target,
  args=(env_factory, i, pipe, sequence_length)) for i, pipe in enumerate(children_pipes)]
  for actor in actors:
    actor.start()
  rewards = []
  dones = []
  for learner_step in range(sequence_length):

    ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *[pipe.recv() for pipe in parent_pipes])
    actions = agent.batched_actor_step(ts.observation)
    for i, pipe in enumerate(parent_pipes):
      pipe.send(actions[i])
    rewards.append(ts.reward)
    dones.append(ts.step_type == dm_env.StepType.LAST)
  for actor in actors:
    actor.join()
    actor.close()
  print(f'Average return: {np.max(returns(rewards, dones))}')
  return np.max(returns(rewards, dones))