# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-np.dot(tx,w)
    e=np.power(e,2)
    MSE=np.mean(e)
    MSE/=2
    return MSE

def compute_mae(y,tx,w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-np.dot(tx,w)
    e=np.absolute(e,2)
    MAE=np.mean(e)
    return MAE