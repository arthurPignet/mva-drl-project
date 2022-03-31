from copy import deepcopy
import dm_env

import multiprocessing as mp
import numpy as np
import jax.numpy as jnp
import jax

import tree
from typing import *

from .utils import actor_target, sequence_of_timesteps_with_action_to_trajectory, returns, my_process
from .plot_utils import pretty_print
from .agents import Agent, A2CAgent, DDPGAgent
from .replay_buffer import BatchedReplayBuffer


def simple_interaction_loop(agent: Agent, environment: dm_env.Environment, max_num_steps: int = 5000) -> None:
    ts = environment.reset()
    for _ in range(max_num_steps):
        if ts.last():
            break

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


def a2c_parallel_interaction_loop(agent: A2CAgent, env_factory: Callable[[], dm_env.Environment],
                                  max_learner_steps: int, sequence_length: int, num_actors: int = 2):
    parent_pipes, children_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
    actors = [mp.Process(target=actor_target, args=(env_factory, i, pipe, max_learner_steps * sequence_length + 1)) for
              i, pipe in enumerate(children_pipes)]
    for actor in actors:
        actor.start()
    timesteps_with_action = []
    for learner_step in range(max_learner_steps):
        timesteps_with_action = timesteps_with_action[-1:]
        while len(timesteps_with_action) < sequence_length + 1:
            ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *[pipe.recv() for pipe in parent_pipes])
            actions = agent.batched_actor_step(ts.observation)
            for i, pipe in enumerate(parent_pipes):
                pipe.send(actions[i])
            timesteps_with_action.append((ts, actions))
        trajectory = sequence_of_timesteps_with_action_to_trajectory(timesteps_with_action)
        logs = agent.learner_step(trajectory)
        if learner_step % 10 == 0:
            print(pretty_print(logs))
    for actor in actors:
        actor.join()


def evaluation_parallel_interaction_loop(agent: A2CAgent,
env_factory: Callable[[],
 dm_env.Environment],
  sequence_length: int,
   num_actors: int = 2):
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
  print(f'Average return: {np.mean(returns(rewards, dones))}')
  for actor in actors:
    actor.join()
    actor.close()


def ddpg_parallel_interaction_loop(agent: DDPGAgent, env_factory: Callable[[], dm_env.Environment], max_learner_steps: int, buffer_size:int = 1000, batch_size: int = 32, num_actors: int = 2, seed = 0):
  plots = dict(iterations = [],
               actions_mean=[],
               value_loss=[],
               policy_loss=[],
               value_mean=[],
               value_target_mean=[],
               mean_reward=[],
               obs=[],
               )
  eps = 1
  rng_key = jax.random.PRNGKey(seed=seed)
  parent_pipes, children_pipes = zip(*[mp.Pipe(duplex=True) for _ in range(num_actors)])
  actors = [mp.Process(target=actor_target, args=(env_factory, np.random.randint(100), pipe, max_learner_steps)) for i, pipe in enumerate(children_pipes)]
  replay = BatchedReplayBuffer(buffer_size)
  for actor in actors:
    actor.start()

  compt = 0
  for learner_step in range(max_learner_steps):
    ts = tree.map_structure(lambda *x: np.stack(x, axis=0), *[pipe.recv() for pipe in parent_pipes])
    if learner_step>0:
      for i in range (len(ts.observation)):
        replay.add(timestep.observation[i],actions[i],ts.reward[i],ts.discount[i],ts.observation[i],ts.step_type[i])
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
    actions = actions + jnp.reshape(my_process(x0=0,paths=actions.shape[0])((0,1))[1],(actions.shape))
    for i, pipe in enumerate(parent_pipes):
      pipe.send(actions[i])
    timestep = deepcopy(ts)

  for actor in actors:
    actor.join()
  return plots