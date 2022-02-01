from .math import tick_to_sqrt_price

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
        self.i = i
        self.sqrt_price = tick_to_sqrt_price(i)

        self.liquidity_net = 0
        self.fee_growth_outside0 = init_fee_growth_global0
        self.fee_growth_outside1 = init_fee_growth_global1

        # sounds like liquidity gross is really for checking whether any
        # liquidity references this tick so we know when to delete it
        # TODO: determine if we need to track this if we don't necessarily care
        #  about optimizing for performance/memory
        self.liquidity_gross = 0

    def update_liquidity(self, liquidity_delta):
        """
        TODO: update documentation

        :param liquidity_delta:
        """
        # TODO: figure out how to update liquidity_gross (if we keep it)
        self.liquidity_net += liquidity_delta

    # TODO: make sure this happens when a tick is crossed during a swap
    def update_fee_growth_outside(self, fee_growth_global):
        """
        TODO: finish documentation

        :param fee_growth_global:
        """
        # formula 6.20 (formulas are the same for each token)
        self.fee_growth_outside0 = fee_growth_global - self.fee_growth_outside0
        self.fee_growth_outside1 = fee_growth_global - self.fee_growth_outside1

    def __repr__(self):
        return f"Tick(i={self.i:,.0f}, price={self.sqrt_price ** 2:,.0f})"
