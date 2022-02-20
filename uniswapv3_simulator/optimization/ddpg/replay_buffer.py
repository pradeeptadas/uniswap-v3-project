import numpy as np


class ReplayBuffer:
    def __init__(self, memory_size=1000000, seed=None):
        self._memory_size = memory_size
        self.reset()
        self.seed(seed=seed)

    @property
    def memory(self):
        return self._memory

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self._memory = []
        self._next_idx = 0

    def add(self, experience):
        if self._next_idx >= len(self):
            self._memory.append(experience)
        else:
            self._memory[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._memory_size

    def sample(self, batch_size):
        return [self._memory[i] for i in
                self._rng.integers(0, len(self), batch_size)]

    def __len__(self):
        return len(self._memory)

    def __repr__(self):
        return f'{type(self).__name__}({len(self):,.0f})'
