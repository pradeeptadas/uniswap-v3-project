import numpy as np
import matplotlib.pyplot as plt
import bisect
from collections import defaultdict
import logging

from .tick import Tick
from .position import PositionMap
from .math import *


logger = logging.getLogger('uniswap-v3.pool')


class SwapState:
    def __init__(self):
        """
        Container class to assist with the swap logic. Based on swap function:
        https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol
        """
        self.step_n = None
        self.amount_remaining = None
        self.amount_calculated = None
        self.sqrt_price = None
        self.tick = None
        self.total_fees = None
        self.fee_growth_global_token_in = None
        self.liquidity = None


class StepComputation:
    def __init__(self):
        """
        Container class to assist with the swap logic. Based on swap function:
        https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol
        """
        self.sqrt_price_start = None
        self.next_tick = None
        self.sqrt_price_next = None
        self.amount_in = None
        self.amount_out = None
        self.fee_amount = None


class Uniswapv3Pool:
    def __init__(self, fee, tick_spacing, init_price,
                 token0_decimals=18, token1_decimals=18):
        """
        TODO: finish documentation
        For simplicity, we assume no protocol fees. Currently, no pools have
        the protocol fees (need to confirm), so this is a reasonable
        simplification.

        :param fee: Transaction fee for the pool in the percentage (not int)
          representation; e.g., fee=0.003 is a 0.3% fee.
        :param tick_spacing:
        :param init_price:
        :param token0_decimals:
        :param token1_decimals:
        """
        self.fee = fee
        self.tick_spacing = tick_spacing

        self.liquidity = 0
        self.sqrt_price = np.sqrt(init_price)  # sqrt(token1 / token0)
        # tick is actually the tick index (i), not the price or Tick object
        # we use this naming convention throughout (e.g., tick_lower)
        self.tick = sqrt_price_to_tick(self.sqrt_price)

        # multipliers for when decimals for each token are not the same
        self.token0_multiplier = 10.0 ** max(token1_decimals - token0_decimals, 0)
        self.token1_multiplier = 10.0 ** max(token0_decimals - token1_decimals, 0)

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

        # additional helper variables (not in Uniswap v3 white
        # paper/implementation)
        self.account_map = defaultdict(set)  # account_id, set of Positions pairs
        # to track actual value of fees earned (not per unit of liquidity
        # like fee_growth_global{0,1})
        self.total_fees_token0 = 0
        self.total_fees_token1 = 0

    @property
    def price(self):
        # p = token1 / token0
        return (
            self.sqrt_price ** 2 *
            (self.token1_multiplier / self.token0_multiplier)
        )

    @property
    def virtual_reserves(self):
        """
        Virtual reserves - values for x and y that allow you to describe the
        contract’s behavior (between two adjacent ticks) as if it followed the
        constant product formula (section 6.2.1 of white paper).

        :return: float, float; virtual reserves for token0, token1
        """
        x = self.liquidity / self.sqrt_price  # formula 6.5
        y = self.liquidity * self.sqrt_price  # formula 6.6

        return x * self.token0_multiplier, y * self.token1_multiplier

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
        logger.debug(
            f'Adding {liquidity_delta:,.6e} to position ({account_id}, '
            f'{tick_lower:,}, {tick_upper:,}).'
        )
        # initialize ticks if needed
        if tick_lower not in self.initd_ticks:
            self._init_tick(tick_lower)
        if tick_upper not in self.initd_ticks:
            self._init_tick(tick_upper)

        # calculate the current fee growth inside the tick range
        # formula 6.19 (formulas are the same for each token)
        fee_growth_inside0 = get_fee_growth_inside(
            self.fee_growth_global0,
            self.tick_map[tick_lower].fee_growth_outside0,
            self.tick_map[tick_upper].fee_growth_outside0,
            self.tick,
            tick_lower,
            tick_upper,
            at_max_tick=(self.tick == self.initd_ticks[-1])
        )
        fee_growth_inside1 = get_fee_growth_inside(
            self.fee_growth_global1,
            self.tick_map[tick_lower].fee_growth_outside1,
            self.tick_map[tick_upper].fee_growth_outside1,
            self.tick,
            tick_lower,
            tick_upper,
            at_max_tick=(self.tick == self.initd_ticks[-1])
        )
        logger.debug(f'token0 fee growth inside position range: {fee_growth_inside0:,.6e}.')
        logger.debug(f'token1 fee growth inside position range: {fee_growth_inside1:,.6e}.')

        # get/initialize the position and update it for liquidity_delta
        position = self.position_map[(account_id, tick_lower, tick_upper)]
        self.account_map[account_id].add(position)
        position.update(liquidity_delta, fee_growth_inside0, fee_growth_inside1,
                        self.token0_multiplier, self.token1_multiplier)

        # update the lower and upper ticks for liquidity_delta
        self.tick_map[tick_lower].update_liquidity(liquidity_delta, False)
        self.tick_map[tick_upper].update_liquidity(liquidity_delta, True)

        # get the sqrt_price for the lower and upper ticks
        sqrt_price_lower = self.tick_map[tick_lower].sqrt_price
        sqrt_price_upper = self.tick_map[tick_upper].sqrt_price

        # if the pool’s current price is within the range of the position, add
        # liquidity_delta to the global liquidity value
        if sqrt_price_lower <= self.sqrt_price <= sqrt_price_upper:
            self.liquidity += liquidity_delta
            logger.debug(f'liquidity_delta added to liquidity.')
            logger.debug(f'Updated liquidity: {self.liquidity:,.6e}.')

        # calculate the amount of token0 and token1 to be deposited into the pool
        # negative values indicate that tokens are removed from the pool
        # formulas 6.29 and 6.30
        delta_token0 = get_delta_token0(
            liquidity_delta,
            self.sqrt_price,
            sqrt_price_lower,
            sqrt_price_upper,
            self.tick,
            tick_lower,
            tick_upper
        ) * self.token0_multiplier

        delta_token1 = get_delta_token1(
            liquidity_delta,
            self.sqrt_price,
            sqrt_price_lower,
            sqrt_price_upper,
            self.tick,
            tick_lower,
            tick_upper
        ) * self.token1_multiplier

        self.token0 += delta_token0
        self.token1 += delta_token1
        logger.debug(f'token0 added to/removed from pool: {delta_token0:,.6e}')
        logger.debug(f'token1 added to/removed from pool: {delta_token1:,.6e}')

        # clear any ticks that are no longer needed
        # ticks will only be removed when removing liquidity
        if liquidity_delta < 0:
            # TODO: Maybe use np.isclose instead of an equality check - would
            #  need to figure out an appropriate tol though.
            if self.tick_map[tick_lower].liquidity_gross == 0:
                self._del_tick(tick_lower)
                logger.debug(
                    f'Tick {tick_lower:,} no longer has liquidity referencing '
                    f'it and was deleted.'
                )
            if self.tick_map[tick_upper].liquidity_gross == 0:
                self._del_tick(tick_upper)
                logger.debug(
                    f'Tick {tick_upper:,} no longer has liquidity referencing '
                    f'it and was deleted.'
                )

        return -delta_token0, -delta_token1

    def swap(self, token, tokens_in):
        """
        TODO: finish documentation

        :param token: 0 or 1, indicating which token was added to the pool
          in exchange for the other.
        :param tokens_in: Number of tokens to add to the pool.
        :return: Number of tokens received. If token0 (token1) is added, returns
          the number of token1 (token0) received.
        """
        assert self.liquidity > 0, f"Cannot swap if pool has 0 liquidity."
        assert token in (0, 1), f"token must be 0 or 1, not {token}."
        assert tokens_in > 0, f"tokens_in must be greater than 0."
        logger.debug(f'Swapping {tokens_in:,.6e} of token{token}.')

        multiplier = self.token0_multiplier if token == 0 else self.token1_multiplier
        tokens_in = tokens_in / multiplier
        logger.debug(f'Scaling token{token} to {tokens_in:,.6e}.')

        # setup the swap state
        state = SwapState()
        state.step_n = 0
        state.amount_remaining = tokens_in
        state.amount_calculated = 0
        state.sqrt_price = self.sqrt_price
        state.tick = self.tick
        state.total_fees = 0
        state.fee_growth_global_token_in = (
            self.fee_growth_global0 if token == 0 else self.fee_growth_global1
        )
        state.liquidity = self.liquidity

        logger.debug(f'Starting tick: {state.tick:,}.')
        logger.debug(f'Starting sqrt_price: {state.sqrt_price:,.6e}.')
        logger.debug(f'Starting liquidity: {state.liquidity:,.6e}.')

        # determine the direction of the swap
        to_the_right = (token == 1)

        # swap within in each range of initialized ticks
        while state.amount_remaining > 0:
            step = StepComputation()
            step.sqrt_price_start = state.sqrt_price

            # The actual implementation swaps one tick at a time, even if the
            # tick is not initialized. As a simplification, we swap across
            # uninitialized and up to each initialized tick. As long as we
            # don't change liquidity, which only happens when we cross an
            # initialized tick, swapping tick-to-tick is the same as swapping
            # across multiple ticks (can try/confirm this using the
            # swap_within_tick function)
            step.next_tick = self._get_next_tick(
                state.tick,
                to_the_right,
                first_call=(state.step_n == 0)
            )
            step.next_tick_sqrt_price = self.tick_map[step.next_tick].sqrt_price
            logger.debug(f'Next initialized tick: {step.next_tick:,}.')
            logger.debug(f'sqrt_price of next tick: {step.next_tick_sqrt_price:,.6e}.')

            # Check if the next tick is the min when moving to the left or
            # max tick when moving to the right. We don't want to cross a
            # min/max tick as there is no liquidity on the other side.
            limit_tick = (
                (step.next_tick == self.initd_ticks[0] and not to_the_right) or
                (step.next_tick == self.initd_ticks[-1] and to_the_right)
            )
            if limit_tick:
                logger.debug('Next tick is the min/max tick initialized.')

            # if the current price is at the price limit (only happens at a
            # limit tick), break as there is no more room to swap
            if limit_tick and (state.sqrt_price == step.next_tick_sqrt_price):
                logger.warning(
                    f'Current price {state.sqrt_price:,.6e} is at a limit '
                    f'point {step.next_tick_sqrt_price:,.6e}. The swap may '
                    f'only be partially executed.'
                )
                break

            # execute the swap up to the next tick sqrt_price
            amount_remaining_less_fees = state.amount_remaining * (1 - self.fee)
            delta_token0, delta_token1, next_sqrt_price = swap_within_tick(
                token,
                amount_remaining_less_fees,
                state.sqrt_price,
                state.liquidity,
                step.next_tick_sqrt_price
            )
            step.amount_in = delta_token0 if token == 0 else delta_token1
            step.amount_out = delta_token1 if token == 0 else delta_token0

            logger.debug('Completed swap within current tick range.')
            logger.debug(f'token{token} in: {step.amount_in:,.6e}.')
            logger.debug(f'token{1 - token} out: {step.amount_out:,.6e}.')

            # update the fee amount calculations
            if next_sqrt_price != step.next_tick_sqrt_price:
                # This fee calculation should be the same as the one below (in
                # the else block), but this one avoids rounding issues when
                # determining amount_remaining.
                step.fee_amount = state.amount_remaining - step.amount_in
            else:
                step.fee_amount = step.amount_in / (1 - self.fee) * self.fee

            state.total_fees += step.fee_amount
            logger.debug(f'Fees earned (in token{token}): {step.fee_amount:.6e}')

            state.amount_remaining -= (step.amount_in + step.fee_amount)
            state.amount_calculated += step.amount_out

            state.sqrt_price = next_sqrt_price
            if state.liquidity > 0:
                state.fee_growth_global_token_in += step.fee_amount / state.liquidity
                logger.debug(
                    f'{step.fee_amount / state.liquidity:,.6e} added to '
                    f'fee_growth_global{token}.'
                )
            logger.debug(f'Amount remaining to swap: {state.amount_remaining:,.6e}.')

            # Cross to the next tick if we've reached the next_tick_sqrt_price
            # in which case the swap was only partially executed within the
            # current tick. We do not cross limit ticks as there is no liquidity
            # on the other side of the tick.
            if not limit_tick and (state.sqrt_price == step.next_tick_sqrt_price):
                logger.debug(f'Crossing tick {step.next_tick:,}.')
                # Add the net liquidity from the crossed tick to the current
                # global liquidity. If we're moving to the left, we interpret
                # the liquidity_net as the opposite sign: row 720 of
                # https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol
                tick_liquidity_net = self.tick_map[step.next_tick].liquidity_net
                if not to_the_right:
                    tick_liquidity_net *= -1
                state.liquidity += tick_liquidity_net
                assert state.liquidity >= 0, 'liquidity cannot be < 0.'
                logger.debug(f'Liquidity added from tick: {tick_liquidity_net:,.6e}')

                # update fee growth outside for the crossed tick
                self.tick_map[step.next_tick].update_fee_growth_outside(
                    state.fee_growth_global_token_in if token == 0 else self.fee_growth_global0,
                    self.fee_growth_global1 if token == 0 else state.fee_growth_global_token_in
                )
                # Due to rounding sqrt_price_to_tick(tick_to_sqrt_price(tick)) != tick
                # for any ticks < 0. Rounding is really only an issue when we
                # are very close to the edges of a tick, which, by definition,
                # happens when crossing a tick. Therefore, instead of selecting
                # the next stick using sqrt_price_to_tick, we just use the
                # already calculated next tick
                state.tick = step.next_tick
            else:
                # We have to use the sqrt_price_to_tick function here as if
                # state.sqrt_price == step.next_tick_sqrt_price, then we didn't
                # make it to the next tick and the current tick could be any
                # tick between the start tick and next tick.
                state.tick = sqrt_price_to_tick(state.sqrt_price)

            # update the swap step
            state.step_n += 1

            logger.debug(f'Current tick: {state.tick:,}.')
            logger.debug(f'Current sqrt_price: {state.sqrt_price:,.6e}.')
            logger.debug(f'Current liquidity: {state.liquidity:,.6e}')

        logger.debug('Swap complete.')
        # once the swap is completed, update pool for the swap and return the
        # amounts the trader has to deposit into the pool (negative value) and
        # receives from the pool (positive value)
        self.sqrt_price = state.sqrt_price
        self.tick = state.tick
        self.liquidity = state.liquidity

        if token == 0:
            amount_token0 = tokens_in - state.amount_remaining
            amount_token1 = state.amount_calculated
            self.fee_growth_global0 = state.fee_growth_global_token_in
            self.total_fees_token0 += state.total_fees

            # unlike Uniswap v2, fees are not added to the pool
            token0_to_pool = amount_token0 - state.total_fees
            token1_to_pool = amount_token1
        else:
            amount_token0 = state.amount_calculated
            amount_token1 = tokens_in - state.amount_remaining
            self.fee_growth_global1 = state.fee_growth_global_token_in
            self.total_fees_token1 += state.total_fees

            token0_to_pool = amount_token0
            # unlike Uniswap v2, fees are not added to the pool
            token1_to_pool = amount_token1 - state.total_fees

        # ensure that tokens have the correct decimal places
        amount_token0 = amount_token0 * self.token0_multiplier
        amount_token1 = amount_token1 * self.token1_multiplier
        token0_to_pool = token0_to_pool * self.token0_multiplier
        token1_to_pool = token1_to_pool * self.token1_multiplier
        logger.debug(f'Scaling token0 to {token0_to_pool:,.6e}.')
        logger.debug(f'Scaling token1 to {token1_to_pool:,.6e}.')

        self.token0 += token0_to_pool
        self.token1 += token1_to_pool

        if token == 0:
            logger.debug(f'Swapped {amount_token0:,.6e} token0 for {-amount_token1:,.6e} token1.')
        else:
            logger.debug(f'Swapped {amount_token1:,.6e} token1 for {-amount_token0:,.6e} token0.')
        logger.debug(f'Total fees earned in token{token}: {state.total_fees:,.6e}.')
        logger.debug(f'Final token0 in pool: {self.token0:,.6e}')
        logger.debug(f'Final token1 in pool: {self.token1:,.6e}')

        return -amount_token0, -amount_token1

    def collect_fees_earned(self, account_id, tick_lower, tick_upper):
        """
        TODO: finish documentation
        TODO: Need a helper function for simply updating tokens_owed{0,1} in
          the position object as they are only updated during the set_position
          function.

        :param account_id:
        :param tick_lower:
        :param tick_upper
        :return:
        """
        logger.debug(
            f'Collecting fees for position ({account_id}, {tick_lower:,}, '
            f'{tick_upper:,}).'
        )
        position = self.position_map[(account_id, tick_lower, tick_upper)]

        amount_token0 = position.tokens_owed0
        amount_token1 = position.tokens_owed1
        logger.debug(f'token0 fees owed: {amount_token0:,.6e}')
        logger.debug(f'token1 fees owed: {amount_token1:,.6e}')

        # limit tokens removed to amounts in pool fees
        if amount_token0 > self.total_fees_token0:
            logger.warning(f'Pool only has {self.total_fees_token0:,.6e} of token0.')
            amount_token0 = self.total_fees_token0

        if amount_token1 > self.total_fees_token1:
            logger.warning(f'Pool only has {self.total_fees_token1:,.6e} of token1.')
            amount_token1 = self.total_fees_token1

        # remove tokens from the pool
        self.total_fees_token0 -= amount_token0
        self.total_fees_token1 -= amount_token1

        # reduce fees owed in the position
        position.tokens_owed0 -= amount_token0
        position.tokens_owed1 -= amount_token1

        logger.debug(f'token0 removed from pool: {amount_token0:,.6e}')
        logger.debug(f'token1 removed from pool: {amount_token1:,.6e}')

        return amount_token0, amount_token1

    def liquidity_curve(self, p):
        """
        TODO: finish documenttion

        :param p:
        :return:
        """
        liquidity_delta = defaultdict(lambda: 0)
        for k, position in self.position_map.items():
            tick_lower = self.tick_map[position.tick_lower]
            tick_upper = self.tick_map[position.tick_upper]

            liquidity_delta[tick_lower.price] += position.liquidity
            liquidity_delta[tick_upper.price] -= position.liquidity

        liquidity_delta = np.array(list(liquidity_delta.items()))
        liquidity_delta = liquidity_delta[liquidity_delta[:, 0].argsort()]

        liquidity_points = liquidity_delta
        liquidity_points[:, 1] = liquidity_delta[:, 1].cumsum()
        liquidity_points = np.insert(liquidity_points, 0, [-np.inf, 0], axis=0)
        assert liquidity_points[-1, -1] == 0, 'Last value of liquidity != 0.'

        return liquidity_points[liquidity_points[:, 0] < p, 1][-1]

    def plot_liquidity_curve(self, ax=None):
        """
        TODO: finish documenttion

        :param ax:
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))

        extra = 0.1
        p_min = tick_to_sqrt_price(self.initd_ticks[0]) ** 2 * (1 - extra)
        p_min = 0 if p_min < 10 else p_min
        p_max = tick_to_sqrt_price(self.initd_ticks[-2]) ** 2 * (1 + extra)
        n_points = 1000

        p = np.linspace(p_min, p_max, n_points)
        l = np.array([self.liquidity_curve(pi) for pi in p])

        ax.plot(p, l, drawstyle='steps')
        ax.set_title('Liquidity Curve')
        ax.set_xlabel('Price (p)')
        ax.set_ylabel('Liquidity (L)')

        return ax

    def _init_tick(self, i):
        """
        Initialize a tick and add it to the tick tracking objects, initd_ticks
        and tick_map. We keep initd_ticks sorted so that we can easily check
        the next tick to each direction).

        :param i: int; tick index.
        """
        assert (i % self.tick_spacing) == 0, (
            'Cannot initialize a tick that is not a multiple of tick_spacing.'
        )
        # initial fee_growth_outside{0,1} values are determined by formula 6.21
        init_fgo0 = get_init_fee_growth_outside(i, self.tick, self.fee_growth_global0)
        init_fgo1 = get_init_fee_growth_outside(i, self.tick, self.fee_growth_global1)

        # initialize the tick and add it to the tracking objects
        tick = Tick(i, init_fgo0, init_fgo1)
        bisect.insort(self.initd_ticks, i)  # insert and keep list sorted
        self.tick_map[i] = tick

    def _del_tick(self, i):
        """
        Delete a tick and remove it from tracking objects, initd_ticks and
        tick_map. This should only be done when a tick no longer has liquidity
        referencing it; this check is done in the set_position function and NOT
        verified here.

        :param i: int; tick index.
        """
        self.initd_ticks.remove(i)
        del self.tick_map[i]

    def _get_next_tick(self, current_tick, to_the_right, first_call=False):
        """
        Get the next initialized tick to the left or right of the current tick.
        Since each tick is a lower bound for a range, the next initialized tick
        may actually be the current tick when moving to the left. We only want
        to get the current tick as the next tick in two cases:
        1. The first time _get_next_tick is called we want to return the current
          tick when moving to the left so that we ensure that we cross the
          current tick (if required).
        2. When we are at a limit tick (i.e., min/max tick), we return the
          current tick as there are no initialized ticks on the other side.

        :param current_tick: int; current tick for which to reference the
          next tick.
        :param to_the_right: bool; whether to get the next tick to the right
          (or, if False, left) of the current tick.
        :param first_call: bool; whether this is the first call to the function.
        :return: int; next initialized tick to the right/left of the current
          tick.
        """
        if to_the_right:
            next_idx = bisect.bisect_right(self.initd_ticks, current_tick)
            # handle max tick
            if next_idx == len(self.initd_ticks):
                return self.initd_ticks[-1]
            else:
                return self.initd_ticks[next_idx]
        else:
            next_idx = bisect.bisect_left(self.initd_ticks, current_tick)
            # Handle cases when the price is above the current tick and the
            # current tick is initialized and may need to be crossed.
            # This would only happen for the first call to _get_next_tick
            # when executing a swap. If current_tick is the max tick, though,
            # we don't want to return the max tick as we need to move to
            # the left.
            if first_call and (current_tick in self.initd_ticks[:-1]):
                return current_tick
            # handle the min tick
            elif next_idx == 0:
                return self.initd_ticks[0]
            else:
                return self.initd_ticks[next_idx - 1]

    def __repr__(self):
        return (
            f"Pool(price={self.price:,.4f}, "
            f"liquidity={self.liquidity:,.2f}, "
            f"fee={self.fee:.2%})"
        )
