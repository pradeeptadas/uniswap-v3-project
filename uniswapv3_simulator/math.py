import numpy as np


def sqrt_price_to_tick(sqrt_price):
    """
    TODO: finish documentation
    See formula 6.8 in the white paper.

    We use the change of base formula to compute the log as numpy doesn't have
    a log function with an arbitrary base.

    :param sqrt_price:
    :return:
    """
    base = np.sqrt(1.0001)
    return int(np.floor(np.log(sqrt_price) / np.log(base)))


def tick_to_sqrt_price(tick):
    """
    TODO: finish documentation
    See formula 6.2 in the white paper.

    :param tick:
    :return:
    """
    return 1.0001 ** (tick / 2)


def token_delta(liquidity_delta,
                sqrt_price, sqrt_price_lower, sqrt_price_upper,
                tick, tick_lower, tick_upper):
    """
    TODO: finish documentation
    Calculate \Delta X, \Delta Y, the amounts of token0 and token1, respectively
    that needs to be contributed to add \Delta L liquidity to the pool. See
    formulas 6.29 and 6.30 in the white paper.

    :param liquidity_delta:
    :param sqrt_price:
    :param tick:
    :param tick_lower:
    :param tick_upper:
    :param sqrt_price_lower:
    :param sqrt_price_upper:
    :return:
    """
    if tick < tick_lower:
        delta_token0 = (
            liquidity_delta
            * (1 / sqrt_price_lower - 1 / sqrt_price_upper)
        )
        delta_token1 = 0

    elif tick < tick_upper:
        delta_token0 = (
            liquidity_delta
            * (1 / sqrt_price - 1 / sqrt_price_upper)
        )
        delta_token1 = (
            liquidity_delta
            * (sqrt_price - sqrt_price_lower)
        )

    else:
        delta_token0 = 0
        delta_token1 = (
            liquidity_delta
            * (sqrt_price_upper - sqrt_price_lower)
        )

    return delta_token0, delta_token1


def init_fee_growth_outside(init_tick, current_tick, fee_growth_global):
    """
    TODO: update documentation

    :param fee_growth_global:
    :param init_tick:
    :param current_tick:
    :return:
    """
    return fee_growth_global if current_tick >= init_tick else 0  # formula 6.21


def fee_growth_above(fee_growth_global, fee_growth_outside,
                     current_tick, tick):
    """
    TODO: update documentation

    :param fee_growth_global:
    :param fee_growth_outside:
    :param current_tick:
    :param tick:
    :return:
    """
    # formula 6.17
    return (
        fee_growth_global - fee_growth_outside if current_tick >= tick
        else fee_growth_outside
    )


def fee_growth_below(fee_growth_global, fee_growth_outside,
                     current_tick, tick):
    """
    TODO: update documentation

    :param fee_growth_global:
    :param fee_growth_outside:
    :param current_tick:
    :param tick:
    :return:
    """
    # formula 6.18
    return (
        fee_growth_outside if current_tick >= tick
        else fee_growth_global - fee_growth_outside
    )


def fee_growth_inside(fee_growth_global,
                      fee_growth_outside_lower, fee_growth_outside_upper,
                      tick, tick_upper, tick_lower):
    """
    TODO: update documentation

    :param fee_growth_global:
    :param fee_growth_outside_lower:
    :param fee_growth_outside_upper:
    :param tick:
    :param tick_upper:
    :param tick_lower:
    :return:
    """
    # formula 6.17
    fa_upper = fee_growth_above(fee_growth_global, fee_growth_outside_upper,
                                tick, tick_upper)
    # formula 6.18
    fb_lower = fee_growth_below(fee_growth_global, fee_growth_outside_lower,
                                tick, tick_lower)

    return fee_growth_global - fb_lower - fa_upper  # formula 6.19

