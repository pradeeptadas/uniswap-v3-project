import numpy as np
import multiprocessing as mp
import itertools
import copy
import logging

from .traders import Uniswapv3Arbitrageur


def convert_to_swap(trade_size, trade_sign, price):
    trade = (trade_size * trade_sign)
    token = 1 if trade > 0 else 0
    tokens_in = trade if trade > 0 else abs(trade) / price

    return token, tokens_in


def arbitrage_to_price(pool, target_price, max_tries=10, fee_multiple=3,
                       calc_profit=True):
    arbitrageur = Uniswapv3Arbitrageur()
    token0, token1, total_profit = 0, 0, 0

    for i in range(max_tries):
        pct_delta = target_price / pool.price - 1
        if abs(pct_delta) < fee_multiple * pool.fee:
            target = target_price
        else:
            target = pool.price * (1 + (pct_delta / 2))

        token, tokens_in = arbitrageur.get_swap(pool, target)
        if (tokens_in is None) or (tokens_in < 1e-12):
            break

        dx, dy = pool.swap(token, tokens_in)
        profit = arbitrageur.calc_profit(-dx, -dy, target_price) if calc_profit else 0

        token0 += dx
        token1 += dy
        total_profit += profit

    if total_profit < 0:
        raise ValueError('Unprofitable arbitrage!')
    if not calc_profit:
        total_profit = np.nan

    return token0, token1, total_profit


def pool_simulation(init_pool, num_trades_dist, trade_size_dist,
                    trade_sign_dist, return_dist,
                    n_innovations=1, n_sims=10000, n_jobs=1,
                    seed=None, calc_profit=False):
    seed_seq = np.random.SeedSequence(seed)
    seeds = seed_seq.generate_state(n_sims)

    iterable = []
    for i in range(n_sims):
        pool = copy.deepcopy(init_pool)
        args = (
            pool, num_trades_dist, trade_size_dist, trade_sign_dist,
            return_dist, n_innovations, seeds[i], calc_profit
        )
        iterable.append(args)

    if n_jobs == 1:
        sim_results = list(itertools.starmap(simulate_trades, iterable))
    elif n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            sim_results = pool.starmap(simulate_trades, iterable)
    else:
        raise ValueError('n_jobs must be >=1.')

    return sim_results


def simulate_trades(pool, num_trades_dist, trade_size_dist, trade_sign_dist,
                    return_dist, n_innovations=1, seed=None, calc_profit=False):
    # suppress logging
    logging.basicConfig(level=logging.ERROR)

    results = {}
    trades = []
    market_prices = []

    seed_seq = np.random.SeedSequence(seed)
    seeds = seed_seq.generate_state(4)
    num_trades_dist.random_state = seeds[0]
    trade_size_dist.random_state = seeds[1]
    trade_sign_dist.random_state = seeds[2]
    return_dist.random_state = seeds[3]

    m = pool.price
    market_prices.append(m)
    min_price = pool.tick_map[pool.initd_ticks[0]].price
    max_price = pool.tick_map[pool.initd_ticks[-1]].price

    for i in range(n_innovations):
        n_liquidity_trades = num_trades_dist.rvs()

        for j in range(n_liquidity_trades):
            # liquidity trade
            trade_size = trade_size_dist.rvs()
            trade_sign = trade_sign_dist.rvs()
            token, tokens_in = convert_to_swap(trade_size, trade_sign, pool.price)
            dx, dy = pool.swap(token, tokens_in)
            trades.append({
                'trade_type': 'liquidity_trade',
                'pool_price': pool.price,
                'dx': dx,
                'dy': dy,
                'profit': np.nan
            })

            # arbitrage towards current price
            dx, dy, profit = arbitrage_to_price(pool, m, calc_profit=calc_profit)
            trades.append({
                'trade_type': 'arbitrage',
                'pool_price': pool.price,
                'dx': dx,
                'dy': dy,
                'profit': profit
            })

        # update the market price
        ret = return_dist.rvs()
        m = m * np.exp(ret)
        m = np.clip(m, min_price, max_price)
        market_prices.append(m)

        # arbitrage towards new market price
        dx, dy, profit = arbitrage_to_price(pool, m, calc_profit=calc_profit)
        trades.append({
            'trade_type': 'arbitrage',
            'pool_price': pool.price,
            'dx': dx,
            'dy': dy,
            'profit': profit
        })

    results['pool'] = pool
    results['trades'] = trades
    results['market_prices'] = market_prices

    return results

