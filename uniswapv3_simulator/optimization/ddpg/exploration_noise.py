import numpy as np


class RandomProcess:
    def __init__(self, seed=None):
        self.seed(seed)

    def sample(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)


class GaussianProcess(RandomProcess):
    def __init__(self, size=None, std=None, seed=None):
        super().__init__(seed=seed)
        self.size = size
        self.std = std

    def sample(self):
        return self._rng.standard_normal(size=self.size) * self.std()


class ConstantNoise(RandomProcess):
    def __init__(self, noise=0, seed=None):
        super().__init__(seed=seed)
        self.noise = noise

    def sample(self):
        return self.noise


class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size=None, std=None, theta=0.15, dt=1e-2, x0=None, seed=None):
        super().__init__(seed=seed)
        self.theta = theta
        self.mu = 0.0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt
            + self.std() * np.sqrt(self.dt)
            * self._rng.standard_normal(size=self.size)
        )
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)