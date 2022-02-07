import numpy as np

from .math import *


def pool_init_price(token0, token1, tick_upper, tick_lower, liquidity_delta):
    """
    TODO: finish documentation

    :param token0:
    :param token1:
    :param tick_upper:
    :param tick_lower:
    :param liquidity_delta: Can get from etherscan.io using the txn hash
      (check the logs).
    :return:
    """
    if (token0 == 0) or (token1 == 0):
        raise ValueError('Tick range does not span the initial price.')
    sqrt_price_lower = tick_to_sqrt_price(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price(tick_upper)

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
                              sqrt_price, check_res=False):
    """
    TODO: finish documentation

    :param token0:
    :param token1:
    :param tick_lower:
    :param tick_upper:
    :param sqrt_price:
    :param check_res:
    :return:
    """
    sqrt_price_lower = tick_to_sqrt_price(tick_lower)
    sqrt_price_upper = tick_to_sqrt_price(tick_upper)
    tick_current = sqrt_price_to_tick(sqrt_price)

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
