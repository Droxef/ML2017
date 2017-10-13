# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    weights=np.linalg.solve(tx.T.dot(tx)+2*tx.shape[0]*lambda_*np.eye(tx.shape[1]),tx.T.dot(y))
    MSE=compute_mse(y,tx,weights)
    return weights,MSE
