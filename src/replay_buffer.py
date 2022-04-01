from .data import Transition

import jax
import jax.numpy as jnp
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object.
        Args:
            batch_size (int): size of each training batch
        """
        self._memory = []
        self._maxlen = buffer_capacity

    def add(self, obs_tm1, action_tm1, reward_t, discount_t, obs_t, done):
        """Add a new transition to memory."""
        if len(self._memory) >= self._maxlen: 
          self._memory.pop(0)  # remove first elem (oldest)

        transition = Transition(
            obs_tm1=obs_tm1,
            action_tm1=action_tm1,
            reward_t=reward_t,
            discount_t=discount_t,
            obs_t=obs_t,
            done=done)
        
        # convert every data into jnp array
        transition = jax.tree_map(jnp.array, transition)

        self._memory.append(transition)

    def sample(self):
        """Randomly sample a transition from memory."""
        assert self._memory, 'replay buffer is unfilled'
        transition_idx = np.random.randint(0, len(self._memory))
        transition = self._memory.pop(transition_idx)
        
        return transition
    

class BatchedReplayBuffer(ReplayBuffer):

      def sample_batch(self, batch_size: int):
        """Randomly sample a batch of experiences from memory."""
        assert len(self._memory) > batch_size, 'Insuficient number of transitions in replay buffer'
        all_transitions = [self.sample() for _ in range(batch_size)] 

        stacked_transitions = []
        for i, _ in enumerate(all_transitions[0]):
          arrays = [t[i] for t in all_transitions]
          arrays = jnp.stack(arrays, axis=0)
          stacked_transitions.append(arrays)

        return Transition(*stacked_transitions)
