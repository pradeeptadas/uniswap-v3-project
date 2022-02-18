import numpy as np
import logging

from .traders import LiquidityTrader, Uniswapv3Arbitrageur
from ..pool import Uniswapv3Pool
from ..utils import set_positions, close_all_positions


logger = logging.getLogger('optimization.environments')


class OneStepEnvironment:
    def __init__(self, init_price, pool_fees, liquidity_fn,
                 mu, sigma, alpha, beta, q,
                 tick_width=1,
                 max_price=None,  # TODO: need to think through this param
                 seed=None):
        self.init_price = init_price
        self.pool_fees = pool_fees

        self.liquidity_fn = liquidity_fn

        # distributions to sample from for each variable
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.q = q

        self.tick_width = tick_width
        self.max_price = max_price
        if self.max_price is None:
            self.max_price = init_price * 3

        self.seed(seed=seed)

        self._state = None
        self._pool = None

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self._pool = Uniswapv3Pool(self.pool_fees, 1, self.init_price)

        mu = self.mu.rvs()
        sigma = self.sigma.rvs()
        alpha = self.alpha.rvs()
        beta = self.beta.rvs()
        q = self.q.rvs()

        state = np.array([
            self.init_price,
            mu,
            sigma,
            alpha,
            beta,
            q
        ])
        self._state = state
        logger.debug('Environment reset.')
        logger.debug(f'Initial state: {state}')

        return state

    def step(self, action):
        logger.debug(f'Action: {action}')
        set_positions(
            self._pool,
            lambda p: self.liquidity_fn(p, *action),
            self.tick_width,
            0,
            self.max_price,
        )
        token0_in = self._pool.token0
        token1_in = self._pool.token1
        start_value = token1_in + token0_in * self.init_price
        logger.debug(f'Initial value of position: {start_value:,.4f}')

        mu = self._state[1]
        sigma = self._state[2]
        alpha = self._state[3]
        beta = self._state[4]
        q = self._state[5]

        info = self.simulate_trades(self._pool, mu, sigma, alpha, beta, q)
        intrinsic_value = info['state']['ending_intrinsic_value']

        tokens = close_all_positions(self._pool)
        token0_out = tokens[0]
        token1_out = tokens[1]
        end_value = token1_out + token0_out * intrinsic_value
        logger.debug(f'Ending value of position: {end_value:,.4f}')

        adv_selection = end_value - start_value
        logger.debug(f'Adverse selection: {adv_selection:,.4f}')

        token0_fees = tokens[2]
        token1_fees = tokens[3]
        total_fees = token1_fees + token0_fees * intrinsic_value
        logger.debug(f'Total fees: {total_fees:,.4f}')

        reward = -((total_fees - adv_selection) ** 2)
        logger.debug(f'Reward: {reward:,.4f}')
        state = None
        done = True

        return state, reward, done, info

    def simulate_trades(self, pool, mu, sigma, alpha, beta, q,
                        max_arb_tries=10, arb_fee_multiple=3):
        results = {}
        results['state'] = {
            'mu': mu,
            'sigma': sigma,
            'alpha': alpha,
            'beta': beta,
            'q': q,
            'start_price': pool.price,
        }

        intrinsic_value = pool.price
        min_price = pool.tick_map[pool.initd_ticks[0]].price
        max_price = pool.tick_map[pool.initd_ticks[-1]].price
        liquidity_trader = LiquidityTrader(beta, q)
        arbitrageur = Uniswapv3Arbitrageur()
        trades = []

        n_liquidity_trades = self._rng.poisson(alpha)
        logger.debug(f'Total liquidity trades: {n_liquidity_trades:,.0f}')
        for _ in range(n_liquidity_trades):
            token, tokens_in = liquidity_trader.get_swap(pool, rng=self._rng)
            logger.debug(f'Liquidity swap: {tokens_in:,.4f} token{token}')
            dx, dy = pool.swap(token, tokens_in)

            trades.append({
                'trade_type': 'liquidity_trade',
                'pool_price': pool.price,
                'dx': dx,
                'dy': dy,
                'profit': np.nan
            })
            logger.debug(f'Pool price after liquidity swap: {pool.price:,.4f}')

            for i in range(max_arb_tries):
                pct_delta = intrinsic_value / pool.price - 1
                if abs(pct_delta) < arb_fee_multiple * pool.fee:
                    price_target = intrinsic_value
                else:
                    price_target = pool.price * (1 + (pct_delta / 2))
                logger.debug(f'Arbitrageur price target: {price_target:,.4f}')

                token, tokens_in = arbitrageur.get_swap(pool, price_target)
                if tokens_in is None:
                    logger.debug(f'No profitable arbitrage found.')
                    break
                logger.debug(f'Arbitrage swap: {tokens_in:,.4f} token{token}')

                dx, dy = pool.swap(token, tokens_in)
                profit = arbitrageur.calc_profit(-dx, -dy, intrinsic_value)
                trades.append({
                    'trade_type': 'arbitrage',
                    'pool_price': pool.price,
                    'dx': dx,
                    'dy': dy,
                    'profit': profit
                })
                logger.debug(f'Pool price after arbitrage swap: {pool.price:,.4f}')

        dt = 1
        z = self._rng.normal(0, 1)
        x = (mu - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * z
        intrinsic_value = intrinsic_value * np.exp(x)
        if intrinsic_value > max_price:
            logger.warning(
                f'Intrinsic value, {intrinsic_value:,.4f}, is greater than the '
                f'max pool price, {max_price:,.4f}.'
            )
            intrinsic_value = max_price
        elif intrinsic_value < min_price:
            logger.warning(
                f'Intrinsic value, {intrinsic_value:,.4f}, is less than the '
                f'min pool price, {min_price:,.4f}.'
            )
            intrinsic_value = min_price
            logger.debug(f'New intrinsic value: {intrinsic_value:,.4f}')

        # TODO: this is the same as above - maybe make it a function?
        for i in range(max_arb_tries):
            pct_delta = intrinsic_value / pool.price - 1
            if abs(pct_delta) < arb_fee_multiple * pool.fee:
                price_target = intrinsic_value
            else:
                price_target = pool.price * (1 + (pct_delta / 2))
            logger.debug(f'Arbitrageur price target: {price_target:,.4f}')

            token, tokens_in = arbitrageur.get_swap(pool, price_target)
            if tokens_in is None:
                logger.debug(f'No profitable arbitrage found.')
                break
            logger.debug(f'Arbitrage swap: {tokens_in:,.4f} token{token}')

            dx, dy = pool.swap(token, tokens_in)
            profit = arbitrageur.calc_profit(-dx, -dy, intrinsic_value)
            trades.append({
                'trade_type': 'arbitrage',
                'pool_price': pool.price,
                'dx': dx,
                'dy': dy,
                'profit': profit
            })
            logger.debug(f'Pool price after arbitrage swap: {pool.price:,.4f}')

        results['trades'] = trades
        results['state']['ending_pool_price'] = pool.price
        results['state']['ending_intrinsic_value'] = intrinsic_value

        return results