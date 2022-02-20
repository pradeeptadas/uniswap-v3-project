import numpy as np


class ConstantScheduler:
    def __init__(self, value):
        self.value = value
        self.step = 0

    def __call__(self):
        value = self.value
        self.step += 1

        return value

    def reset(self):
        self.step = 0


class LinearScheduler:
    def __init__(self, start_value, end_value, steps):
        self.start_value = start_value
        self.end_value = end_value
        self.value = start_value
        self.step = 0

        self.increment = (end_value - start_value) / steps

        if end_value > start_value:
            self.bound = min
        elif end_value < start_value:
            self.bound = max
        else:
            raise ValueError('start_value cannot equal end_value.')

    def __call__(self):
        value = self.value
        self.value = self.bound(self.value + self.increment, self.end_value)
        self.step += 1

        return value

    def reset(self):
        self.step = 0
        self.value = self.start_value


class ExponentialScheduler:
    def __init__(self, start_value, end_value, rate):
        self.start_value = start_value
        self.end_value = end_value
        self.rate = rate
        self.value = start_value
        self.step = 0

        if rate > 1.0:
            self.bound = np.minimum
        elif rate < 1.0:
            self.bound = np.maximum
        else:
            raise ValueError('rate must be >1.0 or <1.0.')

        # if end_value > start_value:
        #     assert rate > 1, 'Rate must be >1.0 one when the end_value > start_value.'
        #     self.bound = np.minimum
        # elif end_value < start_value:
        #     assert rate < 1, 'Rate must be <1.0 one when the end_value < start_value.'
        #     self.bound = np.maximum
        # else:
        #     raise ValueError('start_value cannot equal end_value.')

    def __call__(self):
        value = self.value
        self.value = self.bound(self.value * self.rate, self.end_value)
        self.step += 1

        return value

    def reset(self):
        self.step = 0
        self.value = self.start_value
