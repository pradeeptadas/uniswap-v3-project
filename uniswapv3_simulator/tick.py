import logging

from .math import tick_to_sqrt_price


logger = logging.getLogger('uniswap-v3.tick')


# These are very small and very large numbers, so it is highly unlikely
# we'd ever get to this point. However, this gives us a good proxy for when
# we want to provide liquidity along the entire curve; i.e., [0, infty).
MIN_TICK = -887272
MAX_TICK = -MIN_TICK


def validate_tick(tick):
    assert (MIN_TICK <= tick <= MAX_TICK), (
        f"Tick is outside valid tick range: [{MIN_TICK}, {MAX_TICK}]."
    )


class Tick:
    def __init__(self, i, init_fee_growth_global0, init_fee_growth_global1):
        """
        TODO: finish documentation
        We don't implement secondsOutside, secondsPerLiquidityOutside, and
        tickCumulativeOutside as these values are not used within the contract.

        :param i: int; tick index.
        :param init_fee_growth_global0:
        :param init_fee_growth_global1:
        """
        validate_tick(i)
        self.i = i
        self.sqrt_price = tick_to_sqrt_price(i)

        self.liquidity_net = 0
        self.liquidity_gross = 0
        self.fee_growth_outside0 = init_fee_growth_global0
        self.fee_growth_outside1 = init_fee_growth_global1

        logger.debug(
            f'Tick {self.i:,} initialized (sqrt_price={self.sqrt_price:,.4f}).'
        )

    @property
    def price(self):
        return self.sqrt_price ** 2

    def update_liquidity(self, liquidity_delta, upper):
        """
        TODO: update documentation
        add to liquidity_delta to lower tick's liquidity_net
        subtract liquidity_delta from upper tick's liquidity_net

        :param liquidity_delta:
        :param upper:
        """
        self.liquidity_gross += liquidity_delta
        if upper:
            self.liquidity_net -= liquidity_delta
            logger.debug(f'{liquidity_delta:,.2f} liquidity subtracted from tick {self.i:,}.')
        else:
            self.liquidity_net += liquidity_delta
            logger.debug(f'{liquidity_delta:,.2f} liquidity added to tick {self.i:,}.')

    # TODO: make sure this happens when a tick is crossed during a swap
    def update_fee_growth_outside(self, fee_growth_global0, fee_growth_global1):
        """
        TODO: finish documentation

        :param fee_growth_global0:
        :param fee_growth_global1:
        """
        # formula 6.20 (formulas are the same for each token)
        self.fee_growth_outside0 = fee_growth_global0 - self.fee_growth_outside0
        self.fee_growth_outside1 = fee_growth_global1 - self.fee_growth_outside1

        logger.debug(f'Fee growth outside token0 updated: {self.fee_growth_outside0:,.8f}.')
        logger.debug(f'Fee growth outside token1 updated: {self.fee_growth_outside1:,.8f}.')

    def __repr__(self):
        return f"Tick(i={self.i:,.0f}, price={self.price:,.4f})"
