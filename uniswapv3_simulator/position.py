import numpy as np
import bisect


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

    def update(self, liquidity_delta, fee_growth_inside0, fee_growth_inside1):
        """
        TODO: update documentation

        :param liquidity_delta:
        :param fee_growth_inside0:
        :param fee_growth_inside1:
        """
        # calculate the uncollected/accumulated fees
        # formula 6.28 (formulas are the same for each token)
        uncollected_fees0 = self.liquidity * (fee_growth_inside0 - self.fee_growth_inside0_last)
        uncollected_fees1 = self.liquidity * (fee_growth_inside1 - self.fee_growth_inside1_last)

        # update the position
        self.liquidity += liquidity_delta
        self.fee_growth_inside0_last = fee_growth_inside0
        self.fee_growth_inside1_last = fee_growth_inside1
        self.tokens_owed0 += uncollected_fees0
        self.tokens_owed1 += uncollected_fees1

    def __repr__(self):
        return (
            f"Position(account_id={self.account_id}, "
            f"tick_lower={self.tick_lower:,.0f}, "
            f"tick_upper={self.tick_upper:,.0f})"
        )
