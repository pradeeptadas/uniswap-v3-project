def amount_to_float(amount, decimals):
    if len(amount) <= decimals:
        if amount.startswith('-'):
            sign = -1
            amount = amount[1:]
        else:
            sign = 1
        amount = amount.rjust(decimals, '0')  # pad to decimal digits
        return float('.' + amount) * sign
    else:
        return float(amount[:-decimals] + '.' + amount[-decimals:])