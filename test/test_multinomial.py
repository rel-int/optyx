import math

import numpy as np

from optyx.utils.misc import multinomial


def test_multinomial_does_not_overflow_on_numpy_integers():
    """Regression: numpy integers in the input made the running product a
    numpy int64, which wraps past 2**63 once the total passes twenty and
    turns downstream amplitudes into NaN through a square root. Python
    integers are arbitrary precision, so the coefficient is exact at any
    size."""
    occupation = tuple(np.int64(1) for _ in range(22))
    assert multinomial(occupation) == math.factorial(22)
    assert multinomial([np.int64(20), np.int64(5)]) == math.comb(25, 5)
