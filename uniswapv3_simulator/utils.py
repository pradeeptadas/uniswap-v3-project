import pandas as pd
import numpy as np
from scipy import optimize
import re
import copy
from datetime import timedelta
from collections import defaultdict
import logging

from .pool import Uniswapv3Pool
from uniswapv3_simulator.tick import MIN_TICK, MAX_TICK
from .math import *

logger = logging.getLogger('uniswap-v3.utils')


def pool_init_price(token0, token1, tick_upper, tick_lower, liquidity_delta,
                    token0_decimals, token1_decimals):
    """
    TODO: finish documentation
    :param token0:
    :param token1:
    :param tick_upper:
    :param tick_lower:
    :param liquidity_delta: Can get from etherscan.io using the txn hash
      (check the logs).
    :param token0_decimals:
    :param token1_decimals:
    :return:
    """
    if (token0 == 0) or (token1 == 0):
        raise ValueError('Tick range does not span the initial price.')
    sqrt_price_lower = tick_to_sqrt_price(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price(tick_upper)

    # adjust tokens if different decimal conventions are used
    token0_multiplier = 10.0 ** max(token1_decimals - token0_decimals, 0)
    token1_multiplier = 10.0 ** max(token0_decimals - token1_decimals, 0)

    token0 = token0 / token0_multiplier
    token1 = token1 / token1_multiplier

    # formula 6.29
    sqrt_price = token1 / liquidity_delta + sqrt_price_lower
    # formula 6.30
    calc_token0 = liquidity_delta * (1 / sqrt_price - 1 / sqrt_price_upper)
    # verify that the calculated price satisfies formula 6.30
    assert np.isclose(token0, calc_token0, atol=1e-12, rtol=1e-8), (
        f'Calculated token0 {calc_token0:,.4f} does not match input '
        f'token0 {token0:,.4f}.'
    )

    return sqrt_price ** 2


def solve_for_liquidity_delta(token0, token1, tick_lower, tick_upper,
                              sqrt_price, token0_decimals, token1_decimals,
                              check_res=False):
    """
    TODO: finish documentation
    :param token0:
    :param token1:
    :param tick_lower:
    :param tick_upper:
    :param sqrt_price:
    :param check_res:
    :param token0_decimals:
    :param token1_decimals:
    :return:
    """
    sqrt_price_lower = tick_to_sqrt_price(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price(tick_upper)
    tick_current = sqrt_price_to_tick(sqrt_price)

    # adjust tokens if different decimal conventions are used
    token0_multiplier = 10.0 ** max(token1_decimals - token0_decimals, 0)
    token1_multiplier = 10.0 ** max(token0_decimals - token1_decimals, 0)

    token0 = token0 / token0_multiplier
    token1 = token1 / token1_multiplier

    if tick_current < tick_lower:
        assert token1 == 0, f'Expected token1 to be 0, not {token1:,.4f}.'
        liquidity_delta = token0 / (1 / sqrt_price_lower - 1 / sqrt_price_upper)

    elif tick_current >= tick_upper:
        assert token0 == 0, f'Expected token0 to be 0, not {token0:,.4f}.'
        liquidity_delta = token1 / (sqrt_price_upper - sqrt_price_lower)

    else:
        liquidity_delta = token1 / (sqrt_price - sqrt_price_lower)
        if check_res:
            check = token0 / (1 / sqrt_price - 1 / sqrt_price_upper)
            assert np.isclose(liquidity_delta, check), (
                f'liquidity_delta {liquidity_delta:,.4f} does not match check '
                f'value {check:,.4f}.'
            )

    return liquidity_delta


def set_positions(pool, liquidity_fn, position_width, min_price, max_price,
                  min_liquidity=1, position_id='pos_id', separate_pos=False):
    """
    TODO: finish documentation
    :param pool:
    :param liquidity_fn:
    :param position_width:
    :param min_price:
    :param max_price:
    :param min_liquidity:
    :param position_id:
    :param separate_pos:
    :return:
    """
    tokens = {}
    lower_bounds = np.linspace(
        min_price,
        max_price,
        int((max_price - min_price) / position_width) + 1
    )
    for i, price_lower in enumerate(lower_bounds[:-1]):
        if price_lower == 0:
            tick_lower = sqrt_price_to_tick((price_lower + 1e-8) ** 0.5)
        else:
            tick_lower = sqrt_price_to_tick(price_lower ** 0.5)

        tick_upper = sqrt_price_to_tick((price_lower + position_width) ** 0.5)
        tick_mid = int((tick_lower + tick_upper) / 2)
        price_mid = tick_to_sqrt_price(tick_mid) ** 2
        liquidity = liquidity_fn(price_mid)

        if liquidity >= min_liquidity:
            pid = position_id + str(i + 1) if separate_pos else position_id
            token0, token1 = pool.set_position(
                pid,
                tick_lower,
                tick_upper,
                liquidity
            )
            if pid in tokens:
                tokens[pid]['token0'] += token0
                tokens[pid]['token1'] += token1
            else:
                tokens[pid] = {'token0': token0, 'token1': token1}

    return tokens


def close_all_positions(pool, account_id=None):
    """
    TODO: finish documentation
    :param pool:
    :param account_id:
    :return:
    """
    total_token0 = 0
    total_token1 = 0
    total_fees_token0 = 0
    total_fees_token1 = 0

    if account_id is None:
        position_iter = list(pool.position_map.values())
    else:
        position_iter = list(pool.account_map[account_id])

    for position in position_iter:
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

        total_token0 += token0
        total_token1 += token1
        total_fees_token0 += fees_token0
        total_fees_token1 += fees_token1

    return total_token0, total_token1, total_fees_token0, total_fees_token1


def organize_txns(liquidity, swaps, max_date=None):
    cols = ['tx_hash', 'txn_time', 'liquidity_event']
    liqu_txn = liquidity.loc[:, cols].copy()
    liqu_txn.reset_index(drop=False, inplace=True)
    liqu_txn.rename(columns={'liquidity_event': 'event', 'index': 'orig_idx'},
                    inplace=True)

    cols = ['tx_hash', 'swap_time']
    swap_txn = swaps.loc[:, cols].copy()
    swap_txn.reset_index(drop=False, inplace=True)
    swap_txn.rename(columns={'swap_time': 'txn_time', 'index': 'orig_idx'},
                    inplace=True)
    swap_txn['event'] = 'SWAP'

    all_txn = pd.concat([liqu_txn, swap_txn], axis=0)
    # when there is more than one transaction for a timestamp, we first do
    # liquidity adds, then removes, then swaps, which corresponding the
    # ascending alphabetical order there are quite a few large liquidity
    # transactions that are immediately reversed so doing the adds before the
    # subtracts ensures these are processed correctly
    all_txn = all_txn.sort_values(['txn_time', 'event']).reset_index(drop=True)
    if max_date is not None:
        all_txn.drop(all_txn.index[all_txn['txn_time'] > max_date],
                     axis=0, inplace=True)

    return all_txn


def run_historical_pool(init_price, all_txn, liquidity, swaps,
                        save_freq='D', position_id='generic_LP',
                        checks_on=False, verbose=True,
                        token0_tols={'atol': 1e-12, 'rtol': 1e-8},
                        token1_tols={'atol': 1e-12, 'rtol': 1e-8},
                        liquidity_tols={'atol': 1e-8, 'rtol': 1e-5}):
    fee = liquidity.at[0, 'pool_fee'] / 1e+6
    tick_spacing = liquidity.at[0, 'pool_tick_spacing']
    token0_decimals = liquidity.at[0, 'contract_decimals_token_0']
    token1_decimals = liquidity.at[0, 'contract_decimals_token_1']

    pool = Uniswapv3Pool(fee, tick_spacing, init_price,
                         token0_decimals=token0_decimals,
                         token1_decimals=token1_decimals)
    if verbose:
        print(f'{pool}')

    tx_results = []
    pool_snapshots = {}
    for i, row in all_txn.iterrows():
        logger.debug(f'Transaction {i}.')
        current_time = row['txn_time'].floor(save_freq)
        txn = row['event']
        idx = row['orig_idx']

        if 'LIQUIDITY' in txn:
            token0 = liquidity.at[idx, 'token_0_amount']
            token1 = liquidity.at[idx, 'token_1_amount']
            if txn == 'REMOVE_LIQUIDITY':
                token0 = -1 * token0
                token1 = -1 * token1

            tick_lower = liquidity.at[idx, 'price_tick_lower']
            tick_upper = liquidity.at[idx, 'price_tick_upper']
            liquidity_delta = liquidity.at[idx, 'liquidity']

            if pd.isnull(liquidity_delta):
                liquidity_delta = solve_for_liquidity_delta(
                    token0,
                    token1,
                    tick_lower,
                    tick_upper,
                    pool.sqrt_price,
                    token0_decimals,
                    token1_decimals
                )
            elif checks_on:
                ld_calc = solve_for_liquidity_delta(
                    token0,
                    token1,
                    tick_lower,
                    tick_upper,
                    pool.sqrt_price,
                    token0_decimals,
                    token1_decimals
                )
                assert np.isclose(liquidity_delta, ld_calc, **liquidity_tols), (
                    f'Calculated liquidity_delta {ld_calc:,.12e} does '
                    f'not match liquidity_delta per the data '
                    f'{liquidity_delta:,.12e}.'
                )

            position = pool.position_map[(position_id, tick_lower, tick_upper)]
            # If the liquidity_delta is very, very close to the position's
            # total liquidity, set liquidity_delta to the total liquidity to
            # completely close out the position
            if np.isclose(-position.liquidity, liquidity_delta):
                liquidity_delta = -position.liquidity
            # we also make sure that liquidity_delta cannot be less than the
            # position's total liquidity
            if liquidity_delta < 0:
                if position.liquidity + liquidity_delta < -1:
                    logger.warning(
                        'Transaction could have negative liquidity. Limiting '
                        'liquidity_delta to the current position liquidity.'
                    )
                liquidity_delta = max(liquidity_delta, -position.liquidity)

            token0_calc, token1_calc = pool.set_position(
                position_id,
                tick_lower,
                tick_upper,
                liquidity_delta
            )
        elif txn == 'SWAP':
            token0 = swaps.at[idx, 'token_0_amount']
            token1 = swaps.at[idx, 'token_1_amount']

            token = 0 if token0 > 0 else 1
            tokens_in = token0 if token == 0 else token1

            token0_calc, token1_calc = pool.swap(token, tokens_in)
        else:
            raise ValueError(f'{txn} is not a valid transaction type.')

        if checks_on:
            assert np.isclose(token0, -token0_calc, **token0_tols), (
                f'Transaction {i:,}: token0 output {-token0_calc:,.12e} does '
                f'not match token0 in the data {token0:,.12e}.'
            )
            assert np.isclose(token1, -token1_calc, **token1_tols), (
                f'Transaction {i:,}: token1 output {-token1_calc:,.12e} does '
                f'not match token1 in the data {token1:,.12e}.'
            )

        if i + 1 < all_txn.shape[0]:
            # if the next date is different than the current date, save the
            # pool so we have the starting liquidity curve for the next day
            next_time = all_txn.at[i + 1, 'txn_time'].floor(save_freq)
            if next_time > current_time:
                date_key = next_time.strftime('%Y-%m-%d %H:%M:%S')
                pool_snapshots[date_key] = copy.deepcopy(pool)

        tx_results.append({
            'sqrt_price': pool.sqrt_price,
            'liquidity': pool.liquidity
        })
        if verbose:
            print(f'Completed transaction {i}.')

    return pool_snapshots, tx_results


def get_bin_ticks(price_bins, pool):
    # apply any multiplier to the price bins to get the scaled prices for
    # the pool
    adj_price_bins = [
        price / (pool.token1_multiplier / pool.token0_multiplier)
        for price in price_bins
    ]
    bin_ticks = []
    for price in adj_price_bins:
        if price == 0:
            tick = pool.tick_spacing * int(np.ceil(MIN_TICK / pool.tick_spacing))
            bin_ticks.append(tick)
        elif price == np.inf:
            tick = pool.tick_spacing * int(np.floor(MAX_TICK / pool.tick_spacing))
            bin_ticks.append(tick)
        else:
            tick = sqrt_price_to_tick(price ** 0.5)
            tick = pool.tick_spacing * int(np.round(tick / pool.tick_spacing))
            bin_ticks.append(tick)

    return bin_ticks


def split_position(position_tuple, bin_ticks):
    position_ticks = (position_tuple[1], position_tuple[2])
    if position_ticks[0] < bin_ticks[0]:
        raise ValueError('Position tick lower is less than the min bin tick.')
    if position_ticks[1] > bin_ticks[-1]:
        raise ValueError('Position tick upper is greater than the max bin tick.')

    positions = []
    start = position_ticks[0]
    for i, tick in enumerate(bin_ticks):
        if tick <= start:
            continue

        elif start < tick <= position_ticks[1]:
            end = tick
            bin_start = bin_ticks[i - 1]
            bin_end = tick
            if start == end:
                logger.warning('Cannot set position if tick_lower == tick_upper.')
                continue

            position_id = f'{position_tuple[0]}_bin{i}_{bin_start}_{bin_end}'
            positions.append((position_id, start, end, position_tuple[3]))

            start = tick

        elif tick > position_ticks[1]:
            end = position_ticks[1]
            bin_start = bin_ticks[i - 1]
            bin_end = tick
            if start == end:
                logger.warning('Cannot set position if tick_lower == tick_upper.')
                continue

            position_id = f'{position_tuple[0]}_bin{i}_{bin_start}_{bin_end}'
            positions.append((position_id, start, end, position_tuple[3]))
            break

    return positions


def set_binned_positions(pool, position_map, price_bins):
    bin_ticks = get_bin_ticks(price_bins, pool)

    bin_tokens = defaultdict(lambda: defaultdict(lambda: 0))
    for position_id, position in position_map.items():
        position_tuple = (
            position.account_id,
            position.tick_lower,
            position.tick_upper,
            position.liquidity
        )
        for new_position in split_position(position_tuple, bin_ticks):
            token0, token1 = pool.set_position(*new_position)
            bin_tokens[new_position[0]]['token0'] += token0
            bin_tokens[new_position[0]]['token1'] += token1

    return bin_tokens


def get_hours(timestamp):
    return timestamp.hour + timestamp.minute / 60 + timestamp.second / 360


def calc_irr(cash_flows, times, init_guess=0.01):
    def obj_fn(irr, cash_flows, time):
        return np.sum(cash_flows / (1 + irr[0]) ** time)

    irr, info, ier, msg = optimize.fsolve(obj_fn, np.array([init_guess]),
                                          args=(cash_flows, times),
                                          full_output=True)
    if ier != 1:
        logger.warning(f'Warning: {msg} The solution may not be accurate.')

    return irr[0]


def calc_token_value(token0, token1, price, numeraire_token=1):
    if numeraire_token == 1:
        value = token1 + token0 * price
    elif numeraire_token == 0:
        value = token0 + token1 / price

    return value


def calc_irr_per_bin(start_pool, price_bins, all_txn, liquidity, swaps,
                     period_start, period_end, numeraire_token=1,
                     position_id='generic_LP'):
    token0_decimals = liquidity.at[0, 'contract_decimals_token_0']
    token1_decimals = liquidity.at[0, 'contract_decimals_token_1']
    multiplier = start_pool.token1_multiplier / start_pool.token0_multiplier

    # initialize a new pool object
    pool = Uniswapv3Pool(
        start_pool.fee,
        start_pool.tick_spacing,
        start_pool.price / multiplier,
        token0_decimals=token0_decimals,
        token1_decimals=token1_decimals
    )
    # reset the positions in the start_pool, splitting them by the different
    # price bins
    bin_ticks = get_bin_ticks(price_bins, pool)
    bin_tokens = set_binned_positions(pool, start_pool.position_map, price_bins)

    cash_flows = defaultdict(list)
    times = defaultdict(list)

    # calculate the starting value in the numeraire token for each bin
    for pos_bin, token_dict in bin_tokens.items():
        init_value = calc_token_value(token_dict['token0'], token_dict['token1'],
                                      pool.price, numeraire_token=numeraire_token)
        cash_flows[pos_bin].append(init_value)
        times[pos_bin].append(0)

    # iterate through the transactions
    for i, row in all_txn.iterrows():
        logging.info(f'Transaction {i}.')
        txn = row['event']
        idx = row['orig_idx']
        txn_time = row['txn_time']

        if 'LIQUIDITY' in txn:
            token0 = liquidity.at[idx, 'token_0_amount']
            token1 = liquidity.at[idx, 'token_1_amount']
            if txn == 'REMOVE_LIQUIDITY':
                token0 = -1 * token0
                token1 = -1 * token1

            tick_lower = liquidity.at[idx, 'price_tick_lower']
            tick_upper = liquidity.at[idx, 'price_tick_upper']
            liquidity_delta = liquidity.at[idx, 'liquidity']

            if pd.isnull(liquidity_delta):
                liquidity_delta = solve_for_liquidity_delta(
                    token0,
                    token1,
                    tick_lower,
                    tick_upper,
                    pool.sqrt_price,
                    token0_decimals,
                    token1_decimals
                )

            position_tuple = (position_id, tick_lower, tick_upper, liquidity_delta)
            for new_position in split_position(position_tuple, bin_ticks):
                position = pool.position_map[tuple(new_position[:3])]
                liquidity_delta = new_position[3]

                # If the liquidity_delta is very, very close to the position's
                # total liquidity, set liquidity_delta to the total liquidity
                # to completely close out the position
                if np.isclose(-position.liquidity, liquidity_delta):
                    liquidity_delta = -position.liquidity
                # we also make sure that liquidity_delta cannot be less than the
                # position's total liquidity
                if liquidity_delta < 0:
                    if position.liquidity + liquidity_delta < -1:
                        logging.warning(
                            'Transaction could have negative liquidity. '
                            'Limitting liquidity_delta to the current position '
                            'liquidity.'
                        )
                    liquidity_delta = max(liquidity_delta, -position.liquidity)

                token0, token1 = pool.set_position(
                    new_position[0],
                    new_position[1],
                    new_position[2],
                    liquidity_delta
                )
                # add any liquidity adds/removes to the cash flows for the bin
                cf = calc_token_value(token0, token1, pool.price,
                                      numeraire_token=numeraire_token)
                cash_flows[new_position[0]].append(cf)
                time_since_start = (
                    (txn_time - period_start) /
                    (period_end - period_start)
                )
                assert time_since_start <= 1.0, 'Time since start cannot be >1.0.'
                times[new_position[0]].append(time_since_start)

        elif txn == 'SWAP':
            token0 = swaps.at[idx, 'token_0_amount']
            token1 = swaps.at[idx, 'token_1_amount']

            token = 0 if token0 > 0 else 1
            tokens_in = token0 if token == 0 else token1
            _, _ = pool.swap(token, tokens_in)

        else:
            raise ValueError(f'{txn} is not a valid transaction type.')

    # close all positions for each bin (i.e., account_id)
    for account_id in pool.account_map.keys():
        tokens = close_all_positions(pool, account_id=account_id)
        total_token0, total_token1, total_fees_token0, total_fees_token1 = tokens

        token0 = total_token0 + total_fees_token0
        token1 = total_token1 + total_fees_token1

        # add the cash flows from the position to the cash flows
        cf = calc_token_value(token0, token1, pool.price,
                              numeraire_token=numeraire_token)
        cash_flows[account_id].append(cf)
        times[account_id].append(1.0)

    # calculate the IRR for each position
    irrs = {}
    for (account_id, cfs), (_, ts) in zip(cash_flows.items(), times.items()):
        irr = calc_irr(cfs, ts, init_guess=0.01)
        irrs[account_id] = irr

    return irrs


def calc_all_returns_per_bin(pool_snapshots, all_txn, liquidity, swaps,
                             freq='D', sigma=0.04, numeraire_token=1):
    all_returns = {}
    date_range = pd.date_range(
        min(pool_snapshots.keys()),
        max(pool_snapshots.keys()),
        freq=freq
    )
    # make sure the timestamps have no timezone so we can perform various
    # operations on the timestamps
    all_txn['txn_time'] = all_txn['txn_time'].dt.tz_localize(None)
    liquidity['txn_time'] = liquidity['txn_time'].dt.tz_localize(None)
    swaps['swap_time'] = swaps['swap_time'].dt.tz_localize(None)

    # only iterate through first len(date_range) - 1 items as the last item
    # will not have a full period
    for i, period_start in enumerate(date_range[:-1]):
        period_end = date_range[i + 1]
        start_pool = pool_snapshots[period_start.strftime('%Y-%m-%d %H:%M:%S')]
        price_bins = np.array(
            [0]
            + [start_pool.price * (1 + i * sigma) for i in range(-10, 11)]
            + [np.inf]
        )
        period_idx = (
            (all_txn['txn_time'] >= period_start) &
            (all_txn['txn_time'] < period_end)
        )
        txns = all_txn.loc[period_idx]
        # There are a few days for the DAI-WETH-500 pool where the swaps move
        # to an area of slightly less than 0 liquidity (due to rounding errors),
        # so we wrap this in a try/except. For simplicity, we just skip
        # these days.
        try:
            irrs = calc_irr_per_bin(
                start_pool,
                price_bins,
                txns,
                liquidity,
                swaps,
                period_start,
                period_end,
                numeraire_token=numeraire_token,
                position_id='generic_LP'
            )
            # There are a few days with extremely large IRRs, which appear
            # to be due to LPs creating artificial limit orders. For simplicity,
            # we just skip these days as well.
            add_to_returns = True
            for irr in irrs.values():
                if irr > 1:
                    logger.warning(f'{period_start}: Abnormal IRR {irr:,.2%}.')
                    add_to_returns = False
                    break
            if add_to_returns:
                all_returns[period_start] = irrs

        except AssertionError as e:
            logger.warning(f'{period_start}: {e}')

    return all_returns
