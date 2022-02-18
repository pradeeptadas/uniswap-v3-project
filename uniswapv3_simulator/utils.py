import numpy as np

from .math import *


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


def set_positions(pool, liquidity_fn, tick_size, min_price, max_price,
                  min_liquidity=1, position_id='position_id'):
    """
    TODO: finish documentation

    :param pool:
    :param liquidity_fn:
    :param tick_size:
    :param min_price:
    :param max_price:
    :param min_liquidity:
    :param position_id:
    :return:
    """
    lower_bounds = np.linspace(
        min_price,
        max_price,
        int((max_price - min_price) / tick_size) + 1
    )
    for price_lower in lower_bounds:
        if price_lower == 0:
            tick_lower = sqrt_price_to_tick((price_lower + 1e-8) ** 0.5)
        else:
            tick_lower = sqrt_price_to_tick(price_lower ** 0.5)

        tick_upper = sqrt_price_to_tick((price_lower + tick_size) ** 0.5)
        tick_mid = int((tick_lower + tick_upper) / 2)
        price_mid = tick_to_sqrt_price(tick_mid) ** 2
        liquidity = liquidity_fn(price_mid)

        if liquidity >= min_liquidity:
            pool.set_position(position_id, tick_lower, tick_upper, liquidity)


def close_all_positions(pool):
    """
    TODO: finish documentation

    :param pool:
    :return:
    """
    total_token0 = 0
    total_token1 = 0
    total_fees_token0 = 0
    total_fees_token1 = 0

    position_iter = list(pool.position_map.items())
    for position_id, position in position_iter:
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
        total_fees_token1 += fees_token0

    return total_token0, total_token1, total_fees_token0, total_fees_token1