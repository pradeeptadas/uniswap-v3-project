# TODO: might remove this - not sure if it will be used
import numpy as np
from collections import deque
# TODO: add documentation and logging throughout


class RandomProcess:
    def __init__(self, seed=None):
        self.seed(seed)

    def simulate(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)


class GBMSimulator(RandomProcess):
    def __init__(self, mu, sigma, s0, T, sim_len, seed=None):
        super().__init__(seed)
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0
        self.T = T
        self.sim_len = sim_len
        self.dt = T / (sim_len - 1)

    def simulate(self):
        z = self._rng.normal(0, 1, self.sim_len - 1)
        x = (
            (self.mu - self.sigma ** 2 / 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )
        sim = self.s0 * np.cumprod(np.exp(x))
        sim = np.concatenate([[self.s0], sim], axis=0)

        return sim.reshape(-1, 1)


class ARMAProcess(RandomProcess):
    def __init__(self, phis, thetas, alpha, sim_len, dist='normal',
                 dist_kwargs={'loc': 0, 'scale': 0.01}, burnin=0, seed=None):
        self.phis = np.asarray(phis)
        self.thetas = np.asarray(thetas)
        self.alpha = alpha
        self.sim_len = sim_len
        self.burnin = burnin
        self.p = len(phis)
        self.q = len(thetas)

        self.dist = dist
        self.dist_kwargs = dist_kwargs
        super().__init__(seed=seed)

    def simulate(self):
        y_prev = deque(np.zeros(self.p), self.p)
        eps_prev = deque(np.zeros(self.q), self.q)

        sim = []
        for _ in range(self.sim_len + self.burnin):
            eps = self._rng_dist(**self.dist_kwargs)
            y = (
                self.alpha
                + np.dot(np.array(y_prev), self.phis)
                + np.dot(np.array(eps_prev), self.thetas)
                + eps
            )
            sim.append(y)
            y_prev.appendleft(y)
            eps_prev.appendleft(eps)

        return np.array(sim[self.burnin:]).reshape(-1, 1)

    def seed(self, seed=None):
        super().seed(seed)
        self._rng_dist = getattr(self._rng, self.dist)


class ARMASimulator(ARMAProcess):
    def __init__(self, phis, thetas, alpha, s0, sim_len, dist='normal',
                 dist_kwargs={'loc': 0, 'scale': 0.01}, burnin=0, seed=None):
        super().__init__(phis, thetas, alpha, sim_len - 1, dist=dist,
                         dist_kwargs=dist_kwargs, burnin=burnin, seed=seed)
        self.s0 = s0

    def simulate(self):
        returns = super().simulate().flatten()
        sim = self.s0 * np.cumprod(np.exp(returns))
        sim = np.concatenate([[self.s0], sim], axis=0)

        return sim.reshape(-1, 1)


class GARCHProcess(RandomProcess):
    def __init__(self, mu, alphas, betas, omega, sim_len,
                 dist='standard_normal', dist_kwargs={}, burnin=0, seed=None):
        self.mu = mu
        self.omega = omega
        self.alphas = np.asarray(alphas)
        self.betas = np.asarray(betas)
        self.sim_len = sim_len
        self.burnin = burnin
        self.p = len(alphas)
        self.q = len(betas)

        self.dist = dist
        self.dist_kwargs = dist_kwargs
        super().__init__(seed=seed)

    def simulate(self):
        eps2_prev = deque(np.zeros(self.p), self.p)
        sigma2_prev = deque(np.zeros(self.q), self.q)

        sim = []
        for _ in range(self.sim_len + self.burnin):
            sigma2 = (
                self.omega
                + np.dot(np.array(eps2_prev), self.alphas)
                + np.dot(np.array(sigma2_prev), self.betas)
            )
            e = self._rng_dist(**self.dist_kwargs)
            eps = e * np.sqrt(sigma2)
            y = self.mu + eps

            sim.append(y)
            eps2_prev.appendleft(eps ** 2)
            sigma2_prev.appendleft(sigma2)

        return np.array(sim[self.burnin:]).reshape(-1, 1)

    def seed(self, seed=None):
        super().seed(seed)
        self._rng_dist = getattr(self._rng, self.dist)


class GARCHSimulator(GARCHProcess):
    def __init__(self, mu, alphas, betas, omega, s0, sim_len,
                 dist='standard_normal', dist_kwargs={}, burnin=0, seed=None):
        super().__init__(mu, alphas, betas, omega, sim_len - 1, dist=dist,
                         dist_kwargs=dist_kwargs, burnin=burnin, seed=seed)
        self.s0 = s0

    def simulate(self):
        returns = super().simulate().flatten()
        sim = self.s0 * np.cumprod(np.exp(returns))
        sim = np.concatenate([[self.s0], sim], axis=0)

        return sim.reshape(-1, 1)