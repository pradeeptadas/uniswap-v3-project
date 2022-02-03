import bisect
from collections import defaultdict
import logging
import warnings

from .tick import Tick
from .position import PositionMap
from .math import *


logger = logging.getLogger('uniswap-v3.pool')


class SwapState:
    def __init__(self):
        self.amount_remaining = None
        self.amount_calculated = None
        self.sqrt_price = None
        self.tick = None
        self.total_fees = None
        self.fee_growth_global_token_in = None
        self.liquidity = None


class StepComputation:
    def __init__(self):
        self.sqrt_price_start = None
        self.next_tick = None
        self.initialized = None
        self.sqrt_price_next = None
        self.amount_in = None
        self.amount_out = None
        self.fee_amount = None


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

        # additional helper variables (not in Uniswap v3 white
        # paper/implementation)
        self.account_map = defaultdict(set)  # account_id, set of Positions pairs
        # to track actual value of fees earned (not per unit of liquidity
        # like fee_growth_global{0,1})
        self.total_fees_token0 = 0
        self.total_fees_token1 = 0

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
        logger.debug(f'token0 fee growth inside position range: {fee_growth_inside0:,.8f}.')
        logger.debug(f'token1 fee growth inside position range: {fee_growth_inside1:,.8f}.')

        # get/initialize the position and update it for liquidity_delta
        position = self.position_map[(account_id, tick_lower, tick_upper)]
        self.account_map[account_id].add(position)
        position.update(liquidity_delta, fee_growth_inside0, fee_growth_inside1)

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
            logger.debug(f'Updated liquidity: {self.liquidity:,.2f}.')

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
        )
        delta_token1 = get_delta_token1(
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
        logger.debug(f'token0 added to/removed from pool: {delta_token0:,.4f}')
        logger.debug(f'token1 added to/removed from pool: {delta_token1:,.4f}')

        # clear any ticks that are no longer needed
        # ticks will only be removed when removing liquidity
        if liquidity_delta < 0:
            # TODO: maybe use np.isclose instead of an equality check - would
            #  need to figure out an appropriate tol though
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
        logger.debug(f'Swapping {tokens_in:,.4f} of token{token}.')

        # setup the swap state
        state = SwapState()
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
        logger.debug(f'Starting sqrt_price: {state.sqrt_price:,.4f}.')
        logger.debug(f'Starting liquidity: {state.liquidity:,.2f}.')

        # determine the direction of the swap
        to_the_right = (token == 1)

        while state.amount_remaining > 0:
            step = StepComputation()
            step.sqrt_price_start = state.sqrt_price

            # TODO: add documentation around difference from actual implementation
            #  as we don't move 1 tick at a time, just to the next initialized tick
            step.next_tick = self._get_next_tick(state.tick, to_the_right)
            step.next_tick_sqrt_price = self.tick_map[step.next_tick].sqrt_price
            logger.debug(f'Next initialized tick: {step.next_tick:,}.')
            logger.debug(f'sqrt_price of next tick: {step.next_tick_sqrt_price:,.4f}.')

            # if the current price is at the price limit, break as there is
            # no more room to swap.
            if state.sqrt_price == step.next_tick_sqrt_price:
                logger.warning(
                    f'Current price {state.sqrt_price:,.4f} is at a limit point '
                    f'{step.next_tick_sqrt_price:,.4f}. The swap may only be '
                    f'partially executed.'
                )
                break

            # check if the next tick is the min/max tick
            # we don't want to cross a min/max tick as there is no liquidity
            # on the other side
            if (step.next_tick == self.initd_ticks[0]) or (step.next_tick == self.initd_ticks[-1]):
                logger.debug('Next tick is the min/max tick initialized.')
                okay_to_cross = False
            else:
                okay_to_cross = False

            amount_remaining_less_fees = state.amount_remaining * (1 - self.fee)
            delta_token0, delta_token1, next_sqrt_price = swap_within_tick(
                token,
                amount_remaining_less_fees,
                state.sqrt_price,
                state.liquidity,
                step.next_tick_sqrt_price
            )
            if token == 0:
                step.amount_in = delta_token0
                step.amount_out = delta_token1
            else:
                step.amount_in = delta_token1
                step.amount_out = delta_token0

            logger.debug('Completed swap within current tick range.')
            logger.debug(f'token{token} in: {step.amount_in:,.4f}.')
            logger.debug(f'token{1 - token} out: {step.amount_out:,.4f}.')

            step.fee_amount = step.amount_in / (1 - self.fee) * self.fee
            state.total_fees += step.fee_amount
            logger.debug(f'Fees earned (in token{token}): {step.fee_amount:.4f}')
            # the above calculation should be the same as the original
            # amount_remaining less the amount_in if the swap is completed
            # within the tick range
            if next_sqrt_price != step.next_tick_sqrt_price:
                check_fee_amount = state.amount_remaining - step.amount_in
                assert np.isclose(step.fee_amount, check_fee_amount), (
                    f'Calculated fees do not match: {step.fee_amount:,.4f} '
                    f'vs. {check_fee_amount:,.4f}.'
                )

            state.amount_remaining -= (step.amount_in + step.fee_amount)
            state.amount_calculated += step.amount_out

            state.sqrt_price = next_sqrt_price
            if state.liquidity > 0:
                state.fee_growth_global_token_in += step.fee_amount / state.liquidity
                logger.debug(
                    f'{step.fee_amount / state.liquidity:,.8f} added to '
                    f'fee_growth_global{token}.'
                )
            logger.debug(f'Amount remaining to swap: {state.amount_remaining:,.4f}.')

            # cross to the next tick if we've reached the next_tick_sqrt_price
            # in which case the swap was only partially executed within the
            # current tick
            if okay_to_cross and (state.sqrt_price == step.next_tick_sqrt_price):
                logger.debug(f'Crossing to tick {step.next_tick:,}.')
                # add the net liquidity from the crossed tick to the current
                # global liquidity
                # if we're moving to the left, we interpret the liquidity_net
                # as the opposite sign: row 720 of
                # https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol
                tick_liquidity_net = self.tick_map[step.next_tick].liquidity_net
                if not to_the_right:
                    tick_liquidity_net *= -1
                state.liquidity += tick_liquidity_net
                assert state.liquidity >= 0, 'liquidity cannot be < 0.'
                logger.debug(f'Liquidity added from tick: {tick_liquidity_net:,.2f}')

                # update fee growth outside for the crossed tick
                self.tick_map[step.next_tick].update_fee_growth_outside(
                    state.fee_growth_global_token_in if token == 0 else self.fee_growth_global0,
                    self.fee_growth_global1 if token == 0 else state.fee_growth_global_token_in
                )

            # update the current tick
            state.tick = sqrt_price_to_tick(state.sqrt_price)

            logger.debug(f'Current tick: {state.tick:,}.')
            logger.debug(f'Current sqrt_price: {state.sqrt_price:,.4f}.')
            logger.debug(f'Current liquidity: {state.liquidity:,.2f}')

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

            token0_to_pool = amount_token0 - state.total_fees
            token1_to_pool = amount_token1
        else:
            amount_token0 = state.amount_calculated
            amount_token1 = tokens_in - state.amount_remaining
            self.fee_growth_global1 = state.fee_growth_global_token_in
            self.total_fees_token1 += state.total_fees

            token0_to_pool = amount_token0
            token1_to_pool = amount_token1 - state.total_fees

        self.token0 += token0_to_pool
        self.token1 += token1_to_pool

        logger.debug(f'Total fees earned in token{token}: {state.total_fees:,.4f}.')
        logger.debug(f'Total token0 added to/removed from pool: {token0_to_pool:,.4f}')
        logger.debug(f'Total token1 added to/removed from pool: {token1_to_pool:,.4f}')
        logger.debug(f'Final token0 in pool: {self.token0:,.4f}')
        logger.debug(f'Final token1 in pool: {self.token1:,.4f}')

        return -amount_token0, -amount_token1


        #####################################
        # delta_token0_acc = 0
        # delta_token1_acc = 0
        #
        # amount_remaining = tokens_in
        # fee_amount = 0
        # current_tick = self.tick
        # current_sqrt_price = self.sqrt_price
        # current_liquidity = self.liquidity
        # partial_swap = False
        #
        # # if you add token0 and remove token1, price decreases and the next
        # # tick is to the left
        # # if you add token1 and remove token0, price increases and the next
        # # tick is to the right
        # to_the_right = (token == 1)
        # next_tick = self._get_next_tick(current_tick, to_the_right)
        # if next_tick is None:
        #     warnings.warn(
        #         f'No liquidity to swap {tokens_in:,.4f} of token{token}. '
        #         f'Check the current amount of token{1 - token} in the pool.'
        #     )
        #     return 0, 0
        #
        # while amount_remaining > 0:
        #     logger.debug(f'Current tick: {current_tick:,}.')
        #     logger.debug(f'Current sqrt_price: {current_sqrt_price:,.4f}.')
        #     logger.debug(f'Current liquidity: {current_liquidity:,.2f}.')
        #     logger.debug(f'Next initialized tick: {next_tick:,}')
        #
        #     # determine the price of the next tick (to the left/right depending
        #     # on whether we are adding token0 or token1)
        #     next_tick_sqrt_price = self.tick_map[next_tick].sqrt_price
        #     logger.debug(f'sqrt_price of next tick: {next_tick_sqrt_price:,.4f}')
        #
        #     # execute as much of the swap as possible within the tick
        #     # see section 6.2.3 of the white paper
        #     delta_token0, delta_token1, next_sqrt_price = swap_within_tick(
        #         token,
        #         amount_remaining,
        #         current_sqrt_price,
        #         current_liquidity,
        #         next_tick_sqrt_price
        #     )
        #     logger.debug('Completed swap within current tick.')
        #     logger.debug(f'Delta token0: {delta_token0:,.4f}.')
        #     logger.debug(f'Delta token1: {delta_token1:,.4f}.')
        #     logger.debug(f'Next sqrt_price: {next_sqrt_price:,.4f}.')
        #
        #     delta_token0_acc += delta_token0
        #     delta_token1_acc += delta_token1
        #     if token == 0:
        #         amount_remaining -= delta_token0
        #     else:
        #         amount_remaining -= delta_token1
        #     # we allow for very small negative values as these are due to
        #     # rounding errors and act the same as 0
        #     # -1e-10 was chosen relatively arbitrarily
        #     assert amount_remaining >= -1e-10, (
        #         'Negative amount of tokens remaining after swap.'
        #     )
        #     logger.debug(f'Remaining token{token}s: {amount_remaining:,.4f}.')
        #
        #     # if the amount remaining was not consumed by the swap within the
        #     # tick, we cross the next tick
        #     # see section 6.3.1 of the white paper
        #     if amount_remaining > 0:
        #         current_tick = next_tick
        #         current_sqrt_price = next_sqrt_price
        #         assert sqrt_price_to_tick(current_sqrt_price) == current_tick, (
        #             'Calculated current tick does not match current tick from swap.'
        #         )
        #         # get the next, next tick (the tick to the left/right of the
        #         # previous next_tick, now current_tick)
        #         next_tick = self._get_next_tick(current_tick, to_the_right)
        #         if next_tick is None:
        #             warnings.warn(
        #                 'Not enough liquidity to execute the entire swap.'
        #             )
        #             partial_swap = True
        #             break
        #
        #         # cross the next tick (now current_tick)
        #         # TODO: this check might be wrong... need to revisit
        #         # we check for a tick to the left/right of the tick before
        #         # crossing the tick to ensure that the swap can continue
        #         # we don't want to cross a tick if there is no liquidity on
        #         # the other side
        #         logger.debug(f'Cross the next tick (tick {current_tick:,}).')
        #         # add the net liquidity from the crossed tick to the current
        #         # global liquidity
        #         # if we're moving to the left, we interpret the liquidity_net
        #         # as the opposite sign: row 720 of
        #         # https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol
        #         tick_liquidity_net = self.tick_map[current_tick].liquidity_net
        #         if not to_the_right:
        #             tick_liquidity_net *= -1
        #         current_liquidity += tick_liquidity_net
        #
        #         # update fee growth outside for the crossed tick
        #         self.tick_map[current_tick].update_fee_growth_outside(
        #             self.fee_growth_global0,
        #             self.fee_growth_global1,
        #         )
        #     else:
        #         # break if there is no more amount_remaining to swap
        #         logger.debug('Swap complete.')
        #         break
        #
        # # TODO: need to move the feel logic inside the while loop as fee growth
        # #  global is on a per unit of liquidity basis and liquidity changes as
        # #  cross ticks
        # # Calculate the total tokens the trader must add and how much he/she
        # # receives. Actual tokens in may differ from tokens_in if only part of the
        # # swap was executed due to limited liquidity.
        # # Also, add the fees for the transaction to fee_growth_global{0,1}.
        # # Only fees for tokens swapped are added. If a transaction was only
        # # partially swapped, the fees are adjusted accordingly.
        # if token == 0:
        #     total_delta_token0 = delta_token0_acc / (1 - self.fee)
        #     total_delta_token1 = delta_token1_acc
        #     self._check_swap_values(tokens_in, total_delta_token0, partial_swap)
        #
        #     logger.debug(f'Token0 added by trader: {abs(total_delta_token0):,.4f}')
        #     logger.debug(f'Token1 received by trader: {abs(total_delta_token1):,.4f}')
        #
        #     fees_added = total_delta_token0 * self.fee
        #     self.fee_growth_global0 += fees_added / self.liquidity
        # else:
        #     total_delta_token1 = delta_token1_acc / (1 - self.fee)
        #     total_delta_token0 = delta_token0_acc
        #     self._check_swap_values(tokens_in, total_delta_token1, partial_swap)
        #
        #     logger.debug(f'Token0 received by trader: {abs(total_delta_token0):,.4f}')
        #     logger.debug(f'Token1 added by trader: {abs(total_delta_token1):,.4f}')
        #
        #     fees_added = total_delta_token1 * self.fee
        #     self.fee_growth_global1 += fees_added / self.liquidity
        #
        # logger.debug(f'Total token{token} added to pool fees: {fees_added:,.4f}')
        # logger.debug(f'Total token0 added to/removed from pool: {delta_token0_acc:,.4f}')
        # logger.debug(f'Total token1 added to/removed from pool: {delta_token1_acc:,.4f}')
        #
        # self.sqrt_price = next_sqrt_price
        # self.tick = sqrt_price_to_tick(self.sqrt_price)
        # self.liquidity = current_liquidity
        # self.token0 += delta_token0_acc
        # self.token1 += delta_token1_acc
        # logger.debug(f'Ending sqrt_price: {self.sqrt_price:,.4f}')
        # logger.debug(f'Ending tick: {self.tick:,}')
        # logger.debug(f'Ending liquidity: {self.liquidity:,.2f}')
        # logger.debug(f'Ending token0: {self.token0:,.4f}')
        # logger.debug(f'Ending token1: {self.token1:,.4f}')
        #
        # return -total_delta_token0, -total_delta_token1

    def collect_fees_earned(self):
        # extract the tokens_owed{0,1} from the position (and the pool)
        pass

    def _init_tick(self, i):
        """
        TODO: finish documentation

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
        TODO: finish documentation

        :param i: int; tick index.
        """
        self.initd_ticks.remove(i)
        del self.tick_map[i]

    def _get_next_tick(self, current_tick, to_the_right):
        """
        TODO: finish documentation

        :param current_tick:
        :param to_the_right:
        :return:
        """
        if to_the_right:
            next_idx = bisect.bisect_right(self.initd_ticks, current_tick)
            if next_idx == len(self.initd_ticks):
                return self.initd_ticks[-1]
            else:
                return self.initd_ticks[next_idx]
        else:
            next_idx = bisect.bisect_left(self.initd_ticks, current_tick)
            if next_idx == 0:
                return self.initd_ticks[0]
            else:
                return self.initd_ticks[next_idx - 1]

    @staticmethod
    def _check_swap_values(tokens_in, total_delta_token, partial_swap):
        """
        TODO: finish documentation

        :param tokens_in:
        :param total_delta_token:
        :param partial_swap:
        :return:
        """
        if not partial_swap:
            assert np.isclose(tokens_in, total_delta_token,
                              atol=1e-12, rtol=1e-8), (
                'total_delta_token0 does not match tokens_in for a '
                'complete swap.'
            )
        else:
            assert total_delta_token < tokens_in, (
                'total_delta_token0 should be less than tokens_in for a '
                'partial swap.'
            )

    def __repr__(self):
        return (
            f"Pool(price={self.price:,.4f}, "
            f"liquidity={self.liquidity:,.2f}, "
            f"fee={self.fee:.2%})"
        )
