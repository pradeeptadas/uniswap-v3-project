import numpy as np
import itertools
from collections import defaultdict
import multiprocessing as mp
import logging

from .traders import LiquidityTrader, Uniswapv3Arbitrageur
from .liquidity_fns import sech2_fn
from ..pool import Uniswapv3Pool
from ..math import *
from ..utils import *


logger = logging.getLogger('optimization.environments')


class OneStepEnvironment:
    def __init__(self, init_price, liquidity_bins,
                 fees, mu, sigma, alpha, beta,
                 n_sims_per_step=50, n_jobs=1, seed=None):
        self.init_price = init_price
        self.liquidity_bins = liquidity_bins

        # distributions to sample from for each variable
        self.fees = fees
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self.n_sims_per_step = n_sims_per_step
        self.n_jobs = n_jobs

        self.seed(seed=seed)

        self._obs = None
        self._closed = True

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        fees = self.fees.rvs()
        mu = self.mu.rvs()
        sigma = self.sigma.rvs()
        alpha = self.alpha.rvs()
        beta = self.beta.rvs()
        obs = np.array([fees, mu, sigma, alpha, beta])

        self._obs = obs
        self._closed = False
        logger.debug('Environment reset.')
        logger.debug(f'Initial observation: {obs}')

        return obs

    def step(self, action):
        assert not self._closed, 'Environment is closed.'
        iterable = [
            (self.init_price, self.liquidity_bins, self._obs, action, logger)
            for _ in range(self.n_sims_per_step)
        ]

        # TODO: need to add random seeding here...
        if self.n_jobs == 1:
            sim_results = itertools.starmap(uniswapv3_simulation, iterable)
        elif self.n_jobs > 1:
            with mp.Pool(self.n_jobs) as pool:
                sim_results = pool.starmap(uniswapv3_simulation, iterable)
        else:
            raise ValueError('n_jobs must be >=1.')

        position_returns = defaultdict(list)
        for ret_dict in sim_results:
            for position_id, ret in ret_dict.items():
                position_returns[position_id].append(ret)

        expected_returns = []
        for position_id, returns in position_returns.items():
            exp_ret = np.mean(returns)
            expected_returns.append(exp_ret)
            logger.debug(f'Expected return {position_id}: {exp_ret:,.2%}')

        reward = -np.std(expected_returns)
        logger.debug(f'Reward: {reward:,.4f}')
        done = True

        return self._obs, reward, done, {}

    def close(self):
        self._obs = None
        self._closed = True


class CompetitiveOneStepEnvironment:
    def __init__(self, init_price, liquidity_bins,
                 fees, mu, sigma, alpha, beta,
                 curve_params=(100, 1000, 10000),
                 n_sims_per_step=50, n_jobs=1, seed=None):
        self.init_price = init_price
        self.liquidity_bins = liquidity_bins

        # distributions to sample from for each variable
        self.fees = fees
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self.curve_params = curve_params
        self.n_sims_per_step = n_sims_per_step
        self.n_jobs = n_jobs

        self.seed(seed=seed)

        self._obs = None
        self._closed = True

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        fees = self.fees.rvs()
        mu = self.mu.rvs()
        sigma = self.sigma.rvs()
        alpha = self.alpha.rvs()
        beta = self.beta.rvs()
        obs = np.array([fees, mu, sigma, alpha, beta])

        self._obs = obs
        self._closed = False
        logger.debug('Environment reset.')
        logger.debug(f'Initial observation: {obs}')

        return obs

    def step(self, action):
        assert not self._closed, 'Environment is closed.'
        iterable = [
            (self.init_price, self.liquidity_bins, self.curve_params,
             self._obs, action, logger)
            for _ in range(self.n_sims_per_step)
        ]

        # TODO: need to add random seeding here...
        if self.n_jobs == 1:
            sim_results = itertools.starmap(competitive_simulation, iterable)
        elif self.n_jobs > 1:
            with mp.Pool(self.n_jobs) as pool:
                sim_results = pool.starmap(competitive_simulation, iterable)
        else:
            raise ValueError('n_jobs must be >=1.')

        exp_ret = np.mean(sim_results)
        sigma = np.std(sim_results)
        reward = exp_ret / sigma
        logger.debug(f'Reward (Sharpe ratio): {reward:,.2%}')
        done = True

        return self._obs, reward, done, {}

    def close(self):
        self._obs = None
        self._closed = True


def uniswapv3_simulation(init_price, liquidity_bins, obs, action, logger):
    # TODO: clean up logging here
    import logging
    logging.basicConfig(level=logging.ERROR)

    fee, mu, sigma, alpha, beta = obs
    pool = Uniswapv3Pool(fee, 1, init_price)

    assert len(liquidity_bins) - 1 == len(action), (
        'Number of actions does not match the number of bins.'
    )
    init_values = {}
    for i in range(len(action)):
        account_id = f'position_{i + 1}'
        tick_lower = sqrt_price_to_tick(liquidity_bins[i] ** 0.5)
        tick_upper = sqrt_price_to_tick(liquidity_bins[i + 1] ** 0.5)
        liquidity = action[i]
        assert liquidity > 0, 'Liquidity must be greater than 0 for all bins.'
        token0, token1 = pool.set_position(
            account_id,
            tick_lower,
            tick_upper,
            liquidity
        )
        value = calc_token_value(-token0, -token1, pool.price)
        init_values[account_id] = value
        logger.debug(f'Initial value of {account_id}: {value:,.4f}')

    try:
        info = simulate_trades(pool, mu, sigma, alpha, beta, 0.5, logger)
    except AssertionError as e:
        logger.warning(f'{e} Ending simulation.')
        # TODO: think about whether this is the best way to handle this
        return 9

    intrinsic_value = info['state']['ending_intrinsic_value']
    ending_values = {}
    for position in pool.position_map.values():
        token0, token1 = pool.set_position(
            position.account_id,
            position.tick_lower,
            position.tick_upper,
            -position.liquidity
        )
        fees_token0, fees_token1 = pool.collect_fees_earned(
            position.account_id,
            position.tick_lower,
            position.tick_upper,
        )
        value = calc_token_value(token0 + fees_token0, token1 + fees_token1,
                                 intrinsic_value)
        ending_values[position.account_id] = value
        logger.debug(f'Ending value of {position.account_id}: {value:,.4f}')

    returns = {}
    for account_id in init_values.keys():
        ret = ending_values[account_id] / init_values[account_id] - 1
        returns[account_id] = ret
        logger.debug(f'Return for {account_id}: {ret:,.2%}')

    return returns


def competitive_simulation(init_price, liquidity_bins, curve_params,
                           obs, action, logger):
    # TODO: clean up logging here
    import logging
    logging.basicConfig(level=logging.ERROR)

    fee, mu, sigma, alpha, beta = obs
    pool = Uniswapv3Pool(fee, 1, init_price)

    set_positions(pool, lambda p: sech2_fn(p, *curve_params), 1, 0, 1000,
                  min_liquidity=1, position_id='LP1')

    assert len(liquidity_bins) - 1 == len(action), (
        'Number of actions does not match the number of bins.'
    )
    init_value = 0
    account_id = 'LP2'
    for i in range(len(action)):
        tick_lower = sqrt_price_to_tick(liquidity_bins[i] ** 0.5)
        tick_upper = sqrt_price_to_tick(liquidity_bins[i + 1] ** 0.5)
        liquidity = action[i]
        assert liquidity > 0, 'Liquidity must be greater than 0 for all bins.'
        token0, token1 = pool.set_position(
            account_id,
            tick_lower,
            tick_upper,
            liquidity
        )
        value = calc_token_value(-token0, -token1, pool.price)
        init_value += value
    logger.debug(f'Initial value of {account_id}: {init_value:,.4f}')

    try:
        info = simulate_trades(pool, mu, sigma, alpha, beta, 0.5, logger)
    except AssertionError as e:
        logger.warning(f'{e} Ending simulation.')
        # TODO: think about whether this is the best way to handle this
        return 9

    intrinsic_value = info['state']['ending_intrinsic_value']
    ending_value = 0
    for position in pool.position_map.values():
        if position.account_id == account_id:
            token0, token1 = pool.set_position(
                position.account_id,
                position.tick_lower,
                position.tick_upper,
                -position.liquidity
            )
            fees_token0, fees_token1 = pool.collect_fees_earned(
                position.account_id,
                position.tick_lower,
                position.tick_upper,
            )
            value = calc_token_value(token0 + fees_token0, token1 + fees_token1,
                                     intrinsic_value)
            ending_value += value
    logger.debug(f'Ending value of {account_id}: {ending_value:,.4f}')

    total_return = ending_value / init_value - 1
    logger.debug(f'Return for {account_id}: {total_return:,.2%}')

    return total_return


def simulate_trades(pool, mu, sigma, alpha, beta, q, logger, seed=None,
                    max_arb_tries=10, arb_fee_multiple=3):
    rng = np.random.default_rng(seed)
    results = {'state': {
        'mu': mu,
        'sigma': sigma,
        'alpha': alpha,
        'beta': beta,
        'q': q,
        'start_price': pool.price,
    }}

    intrinsic_value = pool.price
    min_price = pool.tick_map[pool.initd_ticks[0]].price
    max_price = pool.tick_map[pool.initd_ticks[-1]].price
    liquidity_trader = LiquidityTrader(beta, q)
    arbitrageur = Uniswapv3Arbitrageur()
    trades = []

    n_liquidity_trades = rng.poisson(alpha)
    logger.debug(f'Total liquidity trades: {n_liquidity_trades:,.0f}')
    for _ in range(n_liquidity_trades):
        token, tokens_in = liquidity_trader.get_swap(pool, rng=rng)
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
            if (tokens_in is None) or (tokens_in < 1e-8):
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
    z = rng.normal(0, 1)
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
        if (tokens_in is None) or (tokens_in < 1e-8):
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


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.env.__str__())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ScaleWrapper(EnvWrapper):
    def __init__(self, env, obs_scale_fn, action_scale_fn, reward_scale_fn):
        super().__init__(env)
        self._obs_scale_fn = obs_scale_fn
        self._action_scale_fn = action_scale_fn
        self._reward_scale_fn = reward_scale_fn

    def reset(self):
        obs = super().reset()
        scaled_obs = self._obs_scale_fn(obs)
        logger.debug(f'Scaled observation: {scaled_obs}')

        return scaled_obs

    def step(self, action):
        scaled_action = self._action_scale_fn(action)
        logger.debug(f'Scaled action: {scaled_action}')

        obs, reward, done, info = self.env.step(scaled_action)

        scaled_obs = self._obs_scale_fn(obs)
        logger.debug(f'Scaled observation: {scaled_obs}')

        scaled_reward = self._reward_scale_fn(reward)
        logger.debug(f'Scaled reward: {scaled_reward:,.4f}')

        return scaled_obs, scaled_reward, done, info

