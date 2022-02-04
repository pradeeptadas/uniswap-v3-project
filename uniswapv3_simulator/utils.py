import numpy as np

from .math import *


def amount_to_float(amount, decimals):
    """
    TODO: finish documentation

    :param amount:
    :param decimals:
    :return:
    """
    if len(amount) <= decimals:
        if amount.startswith('-'):
            sign = -1
            amount = amount[1:]
        else:
            sign = 1
        amount = amount.rjust(18, '0')  # pad to 18 digits
        return float('.' + amount) * sign
    else:
        return float(amount[:-18] + '.' + amount[-18:])


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
