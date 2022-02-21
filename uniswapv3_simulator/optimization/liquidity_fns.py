import numpy as np


def sech2_fn(p, mu, s, c):
    # sech = 1 / cosh
    return c * (1 / np.cosh((p - mu) / s))**2