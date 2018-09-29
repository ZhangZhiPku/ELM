"""
    This file defined some useful tool method for processing input data.
"""

import numpy as np
from sklearn.preprocessing import scale


def binarilize_classified(classified):
    return np.array([0 if _ <= 0.5 else 1 for _ in classified])


def standard_scaling(dataframe):
    return scale(dataframe, axis=0)
