import logging

from .math import *


logger = logging.getLogger('uniswap-v3.position')


class PositionMap(dict):
    """
    TODO: update documentation
    General idea is from:
    https://stackoverflow.com/questions/25951966/python-defaultdict-with-non-default-argument
    """
    def __missing__(self, key):
        new_position = self[key] = Position(*key)
        return new_position


class Position:
    def __init__(self, account_id, tick_lower, tick_upper):
        """
        TODO: update documentation

        :param account_id:
        :param tick_lower:
        :param tick_upper:
        """
        self.account_id = account_id
        self.tick_lower = tick_lower
        self.tick_upper = tick_upper

        # all values are initialized to 0; see
        # https://github.com/Uniswap/v3-core/blob/main/contracts/libraries/Position.sol
        # default value for uint{256,128} is 0:
        # https://docs.soliditylang.org/en/develop/control-structures.html#default-value
        self.liquidity = 0
        self.fee_growth_inside0_last = 0
        self.fee_growth_inside1_last = 0

        # tokens available to withdraw from the position
        self.tokens_owed0 = 0
        self.tokens_owed1 = 0

        logger.debug(
            f'Position {self.account_id}, {self.tick_lower:,}, '
            f'{self.tick_upper:,} initialized.'
        )

    def update(self, liquidity_delta, fee_growth_inside0, fee_growth_inside1,
               token0_multiplier, token1_multiplier):
        """
        TODO: update documentation

        :param liquidity_delta:
        :param fee_growth_inside0:
        :param fee_growth_inside1:
        """
        # TODO: this has some issues when the current tick is at the end of
        #  the initialized tick range
        #  probably stems from the tick crossing logic (i.e., when to actually
        #  cross a tick)
        # calculate the uncollected/accumulated fees
        uncollected_fees0 = get_uncollected_fees(
            self.liquidity,
            fee_growth_inside0,
            self.fee_growth_inside0_last
        ) * token0_multiplier

        uncollected_fees1 = get_uncollected_fees(
            self.liquidity,
            fee_growth_inside1,
            self.fee_growth_inside1_last
        ) * token1_multiplier

        logger.debug(f'token0 uncollected fees: {uncollected_fees0:,.6e}.')
        logger.debug(f'token1 uncollected fees: {uncollected_fees1:,.6e}.')

        new_liquidity = self.liquidity + liquidity_delta
        assert new_liquidity >= 0, "A position's liquidity cannot be < 0."

        # update the position
        self.liquidity = new_liquidity
        self.fee_growth_inside0_last = fee_growth_inside0
        self.fee_growth_inside1_last = fee_growth_inside1
        self.tokens_owed0 += uncollected_fees0
        self.tokens_owed1 += uncollected_fees1

        logger.debug(f'Updated position liquidity: {self.liquidity:,.6e}.')
        logger.debug(f'Total token0 owed: {self.tokens_owed0:,.6e}.')
        logger.debug(f'Total token1 owed: {self.tokens_owed1:,.6e}.')

    def __repr__(self):
        return (
            f"Position(account_id={self.account_id}, "
            f"tick_lower={self.tick_lower:,.0f}, "
            f"tick_upper={self.tick_upper:,.0f}, "
            f"liquidity={self.liquidity:,.4f})"
        )

    # defining hash and equal so that we can use the Position class with
    # Python sets
    def __hash__(self):
        return hash((self.account_id, self.tick_lower, self.tick_upper))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.__hash__() == other.__hash__()
        else:
            return False
