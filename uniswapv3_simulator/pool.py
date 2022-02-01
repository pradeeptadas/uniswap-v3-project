import bisect
from .tick import Tick
from .position import PositionMap
from .math import *


class Uniswapv3Pool:
    def __init__(self, fee, tick_spacing, init_price):
        """
        TODO: finish documentation
        For simplicity, we assume no protocol fees. Currently, no pools have
        the protocol fees (need to confirm), so this is a reasonable
        simplification.

        :param fee: Transaction fee for the pool in the percentage (not int)
          representation; e.g., fee=0.003 is a 0.3% fee.
        :param tick_spacing:
        :param init_price:
        """
        self.fee = fee
        self.tick_spacing = tick_spacing

        self.liquidity = 0
        self.sqrt_price = np.sqrt(init_price)  # sqrt(token1 / token0)
        # tick is actually the tick index (i), not the price or Tick object
        # we use this naming convention throughout (e.g., tick_lower)
        self.tick = sqrt_price_to_tick(self.sqrt_price)  # TODO: does this need to be initialized?

        # Per page 6 of the white paper: Total amount of fees that have been
        # earned per unit of virtual liquidity (L), over the entire history of
        # the contract. You can think of them as the total amount of fees that
        # would have been earned by 1 unit of unbounded liquidity that was
        # deposited when the contract was first initialized.
        self.fee_growth_global0 = 0
        self.fee_growth_global1 = 0

        self.token0 = 0  # x in the white paper
        self.token1 = 0  # y in the white paper

        # slightly different mechanism for tracking initialized ticks than in
        # the actual implementation/white paper
        self.initd_ticks = []  # sorted list of initialized tick indices
        self.tick_map = {}  # tick index, Tick object pairs

        # (account_id, tick_lower, tick_upper), Position object pairs
        self.position_map = PositionMap()

    @property
    def price(self):
        return self.sqrt_price ** 2  # p = token1 / token0

    @property
    def virtual_reserves(self):
        """
        TODO: finish documentation

        :return:
        """
        x = self.liquidity / self.sqrt_price  # formula 6.5
        y = self.liquidity * self.sqrt_price  # formula 6.6

        return x, y

    def set_position(self, account_id, tick_lower, tick_upper, liquidity_delta):
        """
        TODO: update documentation

        :param account_id: Unique ID for the position.
        :param tick_lower:
        :param tick_upper:
        :param liquidity_delta: Amount of liquidity to add or remove from the
          position. Positive (negative) values indicate adding (removing)
          liquidity.
        :return: Amount of token0 and token1 deposited (negative values) or
          received (positive values) to update the position. Does NOT include
          any fees earned when removing liquidity from the position.
        """
        # initialize ticks if needed
        if tick_lower not in self.initd_ticks:
            self._init_tick(tick_lower)
        if tick_upper not in self.initd_ticks:
            self._init_tick(tick_upper)

        # calculate the current fee growth inside the tick range
        # formula 6.19 (formulas are the same for each token)
        fee_growth_inside0 = fee_growth_inside(
            self.fee_growth_global0,
            self.tick_map[tick_lower].fee_growth_outside0,
            self.tick_map[tick_upper].fee_growth_outside0,
            self.tick,
            tick_upper,
            tick_lower
        )
        fee_growth_inside1 = fee_growth_inside(
            self.fee_growth_global1,
            self.tick_map[tick_lower].fee_growth_outside1,
            self.tick_map[tick_upper].fee_growth_outside1,
            self.tick,
            tick_upper,
            tick_lower
        )

        # get/initialize the position and update it for liquidity_delta
        position = self.position_map[(account_id, tick_lower, tick_upper)]
        position.update(liquidity_delta, fee_growth_inside0, fee_growth_inside1)

        # update the lower and upper ticks for liquidity_delta
        self.tick_map[tick_lower].update_liquidity(liquidity_delta)
        self.tick_map[tick_upper].update_liquidity(liquidity_delta)

        # get the sqrt_price for the lower and upper ticks
        sqrt_price_lower = self.tick_map[tick_lower].sqrt_price
        sqrt_price_upper = self.tick_map[tick_upper].sqrt_price

        # if the pool’s current price is within the range of the position, add
        # liquidity_delta to the global liquidity value
        if sqrt_price_lower <= self.sqrt_price <= sqrt_price_upper:
            self.liquidity += liquidity_delta

        # calculate the amount of token0 and token1 to be deposited into the pool
        # negative values indicate that tokens are removed from the pool
        # formulas 6.29 and 6.30
        delta_token0, delta_token1 = token_delta(
            liquidity_delta,
            self.sqrt_price,
            sqrt_price_lower,
            sqrt_price_upper,
            self.tick,
            tick_lower,
            tick_upper
        )
        self.token0 += delta_token0
        self.token1 += delta_token1

        return -delta_token0, -delta_token1

    def swap(self, token, tokens_in):
        """
        TODO: finish documentation

        :param token: 0 or 1, indicating which token was added to the pool
          in exchange for the other.
        :param tokens_in: Number of tokens to add to the pool
        :return: Number of tokens received. If token0 (token1) is added, returns
          the number of token1 (token0) received.
        """
        assert token in (0, 1), f"token must be 0 or 1, not {token}."

        # TODO: Current logic only implements swapping within a tick.
        #  Need to update so that we can cross ticks (see section 6.3.1).
        #  This function needs some work...

        # calculate the amount to increment fee_growth_global
        delta_fgg = tokens_in * self.fee  # formula 6.9 without protocol fees

        # There is probably a better way to calculate the below. Might be
        # worth looking into the actual sqrt price math:
        # https://github.com/Uniswap/v3-core/blob/main/contracts/libraries/SqrtPriceMath.sol
        if token == 0:
            # increment fee_growth_global
            self.fee_growth_global0 += delta_fgg

            delta_token0 = tokens_in * (1 - self.fee)  # formula 6.11

            # TODO: move this section to math.py?
            delta_sqrt_price_inv = delta_token0 / self.liquidity  # formula 6.15
            next_sqrt_price_inv = (1 / self.sqrt_price) + delta_sqrt_price_inv
            next_sqrt_price = 1 / next_sqrt_price_inv
            delta_sqrt_price = next_sqrt_price - self.sqrt_price

            delta_token1 = delta_sqrt_price * self.liquidity  # formula 6.14

            self.sqrt_price = next_sqrt_price
            self.tick = sqrt_price_to_tick(self.sqrt_price)

            return delta_token1
        else:
            # increment fee_growth_global
            self.fee_growth_global1 += delta_fgg

            delta_token1 = tokens_in * (1 - self.fee)  # formula 6.11

            # TODO: move this section to math.py?
            delta_sqrt_price = delta_token1 / self.liquidity  # formula 6.13
            next_sqrt_price = self.sqrt_price + delta_sqrt_price
            delta_sqrt_price_inv = (1 / next_sqrt_price) - (1 / self.sqrt_price)

            delta_token0 = delta_sqrt_price_inv * self.liquidity  # formula 6.16

            self.sqrt_price = next_sqrt_price
            self.tick = sqrt_price_to_tick(self.sqrt_price)

            return delta_token0

    def _init_tick(self, i):
        """
        TODO: finish documentation

        :param i: int; tick index.
        """
        assert (i % self.tick_spacing) == 0, (
            'Cannot initialize a tick that is not a multiple of tick_spacing.'
        )
        # initial fee_growth_outside{0,1} values are determined by formula 6.21
        init_fgo0 = init_fee_growth_outside(i, self.tick, self.fee_growth_global0)
        init_fgo1 = init_fee_growth_outside(i, self.tick, self.fee_growth_global1)

        # initialize the tick and add it to the tracking objects
        tick = Tick(i, init_fgo0, init_fgo1)
        bisect.insort(self.initd_ticks, i)  # insert and keep list sorted
        self.tick_map[i] = tick

    # TODO: Determine if we need this. If we do keep this, we'll want to update
    #  the set_position function to remove ticks that are no longer referenced
    #  by any positions.
    def _del_tick(self, i):
        self.initd_ticks.remove(i)
        del self.tick_map[i]

    def __repr__(self):
        return (
            f"Pool(price={self.price:,.2f}, "
            f"liquidity={self.liquidity:,.2f}, "
            f"fee={self.fee:.2%})"
        )
