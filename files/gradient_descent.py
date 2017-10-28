# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
from costs import sigmoid

def compute_gradient(y, tx, w):
    """Compute the gradient mse."""
    e=y-np.dot(tx,w)
    grad=-1/y.shape[0]*np.dot(np.transpose(tx),e)
    return grad

def compute_sig_gradient(y,tx,w):
    """ correct the output and calculate the gradient of logistic function """
    y=(y+1)/2
    return tx.T.dot(sigmoid(tx.dot(w))-y)