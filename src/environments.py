from acme import specs
import chex
import dm_env
import gym
import numpy as np


def inverted_pendulum_env_factory(seed: int, for_evaluation: bool = False) -> dm_env.Environment:
    del seed
    return InvertedPendulumEnv(for_evaluation=for_evaluation)

def reacher_env_factory(seed: int, for_evaluation: bool = False) -> dm_env.Environment:
    del seed
    return ReacherEnv(for_evaluation=for_evaluation)


class CustomEnv(dm_env.Environment):
    def __init__(self, name: str, for_evaluation: bool) -> None:
        self._env = gym.make(name)
        self._for_evaluation = for_evaluation
        if self._for_evaluation:
            self.screens = []

    def step(self, action: chex.ArrayNumpy) -> dm_env.TimeStep:
        new_obs, reward, done, _ = self._env.step(action)
        if self._for_evaluation:
            self.screens.append(self._env.render(mode='rgb_array'))
        if done:
            return dm_env.termination(reward, new_obs)
        return dm_env.transition(reward, new_obs)

    def reset(self) -> dm_env.TimeStep:
        obs = self._env.reset()
        if self._for_evaluation:
            self.screens.append(self._env.render(mode='rgb_array'))
        return dm_env.restart(obs)

    def close(self) -> None:
        self._env.close()


class InvertedPendulumEnv(CustomEnv):
    def __init__(self, for_evaluation: bool):
        CustomEnv.__init__(self, "Pendulum-v1", for_evaluation)

    @staticmethod
    def observation_spec() -> specs.BoundedArray:
        return specs.BoundedArray(shape=(3,), minimum=-8., maximum=8., dtype=np.float32)

    @staticmethod
    def action_spec() -> specs.BoundedArray:
        return specs.BoundedArray(shape=(1,), minimum=-2., maximum=2., dtype=np.float32)



class ReacherEnv(CustomEnv):
    def __init__(self, for_evaluation: bool):
        CustomEnv.__init__(self, "Reacher-v2", for_evaluation)

    @staticmethod
    def observation_spec() -> specs.BoundedArray:
        return specs.BoundedArray(shape=(11,), minimum=-np.inf, maximum=np.inf, dtype=np.float32)

    @staticmethod
    def action_spec() -> specs.BoundedArray:
        return specs.BoundedArray(shape=(2,), minimum=-1., maximum=1., dtype=np.float32)
