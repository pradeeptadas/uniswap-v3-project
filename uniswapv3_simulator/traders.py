# TODO: might move this to a separate packege if it doesn't use the other
#  Uniswap code
import numpy as np
from scipy import optimize
# TODO: add documentation and logging throughout


class LiquidityTrader:
    def __init__(self, beta, p, seed=None):
        self._beta = beta
        self._p = p
        self.seed(seed=seed)

    def get_swap(self, pool):
        trade_sizes = self._rng.exponential(self._beta)
        trade_signs = self._rng.choice([1, -1], p=[self._p, 1 - self._p])
        trade = (trade_sizes * trade_signs)

        token = 1 if trade > 0 else 0
        tokens_in = trade if trade > 0 else abs(trade) / pool.price

        return token, tokens_in

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)


class Arbitrageur:
    def __init__(self):
        self.trades = {}
        self.total_profit = 0

    def get_swap(self, pool, true_price):
        raise NotImplementedError

    def calc_profit(self, delta_x, delta_y, conversion_price, save=True):
        # profit is calculated in token1, the numeraire asset
        if (delta_y > 0) and (delta_x < 0):
            profit = (-delta_x) * conversion_price - delta_y
        elif (delta_y < 0) and (delta_x > 0):
            profit = (-delta_y) - delta_x * conversion_price
        else:
            raise ValueError('delta_x and delta_y must be opposite signs.')

        if save:
            self.trades[(delta_x, delta_y)] = profit
            self.total_profit += profit

        return profit


class Uniswapv3Arbitrageur(Arbitrageur):
    @staticmethod
    def _dy(q1, q0, alpha, beta):
        return alpha * (q1 - q0) + beta / 2 * (q1 ** 2 - q0 ** 2)

    @staticmethod
    def _q1(dy, q0, alpha, beta):
        return (-alpha + np.sqrt((alpha + beta * q0) ** 2 + 2 * beta * dy)) / beta

    @staticmethod
    def _dx(dy, q0, q1):
        return -dy / (q1 * q0)

    def get_swap(self, pool, true_price):
        if pool.price == true_price:
            return None, None

        m = true_price
        q0 = pool.sqrt_price
        gamma = pool.fee

        # alpha, beta for the secant line between the liquidity at the
        # current sqrt_price and the liquidity at the true sqrt_price, which the
        # price should move towards
        beta = (
            (pool.liquidity_curve(m) - pool.liquidity_curve(q0 ** 2))
            / (m ** 0.5 - q0)
        )
        alpha = pool.liquidity_curve(q0 ** 2) - beta * q0

        if beta == 0:
            # if beta == 0, then we expect liquidity to be constant in the range
            # and we use the simple arbitrage formulas derived for Uniswap v2
            x, y = pool.virtual_reserves
            k = alpha ** 2

            if m > pool.price:
                dy = np.sqrt(k * m * (1 - gamma)) - y
                dx = np.sqrt(k) / np.sqrt(m * (1 - gamma)) - x
                if dy <= 1e-12:
                    return None, None
            else:
                dy = np.sqrt(k * m) / np.sqrt((1 - gamma)) - y
                dx = np.sqrt(k * (1 - gamma)) / np.sqrt(m) - x
                if dx <= 1e-12:
                    return None, None

        else:
            def _obj_fn(dy, m, q0, gamma, alpha, beta):
                dy = dy[0]
                q1 = self._q1(dy, q0, alpha, beta)
                dx = self._dx(dy, q0, q1)

                # total profit from arbitrage in token1
                if m > q0 ** 2:
                    # if true price > current price, dy > 0 and fees are
                    # paid in token1
                    return -((-dx) * m - dy / (1 - gamma))
                else:
                    # if true price > current price, dy > 0 and fees are
                    # paid in token0
                    return -((-dy) - dx * m / (1 - gamma))

            init_dy = self._dy(m ** 0.5, q0, alpha, beta)
            if m > pool.price:
                # to move the current price to the true price dy > 0
                constraint = {'type': 'ineq', 'fun': lambda x: x}
            else:
                # to move the current price to the true price dy < 0
                constraint = {'type': 'ineq', 'fun': lambda x: -x}

            res = optimize.minimize(
                _obj_fn,
                np.array([init_dy]),
                args=(m, q0, gamma, alpha, beta),
                constraints=constraint
            )
            # TODO: keep an eye on this as it may have to change (e.g., return 0)
            assert res.success, 'Optimization failed.'
            if -res.fun < 0:
                return None, None

            dy = res.x[0]
            q1 = self._q1(dy, q0, alpha, beta)
            dx = self._dx(dy, q0, q1)
            if (abs(dy) <= 1e-12) or (abs(dx) <= 1e-12):
                return None, None

        token = 1 if dy > 0 else 0
        tokens_in = dy if dy > 0 else dx
        tokens_in = tokens_in / (1 - gamma)

        return token, tokens_in


class Uniswapv2Arbitrageur(Arbitrageur):
    def get_swap(self, pool, true_price):
        if pool.price == true_price:
            return None, None

        m = true_price
        gamma = pool.fee
        x, y = pool.virtual_reserves
        k = pool.liquidity_curve(pool.price) ** 2

        if m > pool.price:
            dy = np.sqrt(k * m * (1 - gamma)) - y
            dx = np.sqrt(k) / np.sqrt(m * (1 - gamma)) - x
            if dy <= 1e-12:
                return None, None
        else:
            dy = np.sqrt(k * m) / np.sqrt((1 - gamma)) - y
            dx = np.sqrt(k * (1 - gamma)) / np.sqrt(m) - x
            if dx <= 1e-12:
                return None, None

        token = 1 if dy > 0 else 0
        tokens_in = dy if dy > 0 else dx
        tokens_in = tokens_in / (1 - gamma)

        return token, tokens_in
