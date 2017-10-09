# -*- coding: utf-8 -*-
"""Find least_squares"""
import numpy as np
from costs import *

def least_squares(y, tx):
    """calculate the least squares solution."""
    if len(y)!=tx.shape[0]:
        raise ValueError("y and tx must have the same length")
   # try:
     #   np.linalg.inv(tx.transpose().dot(tx))
    #except:
        #raise ValueError("tx has no pseudo-inverse")
    #pseudoInv=np.linalg.inv(tx.transpose().dot(tx)).dot(tx.transpose())
   # weight=pseudoInv.dot(y)
    weight = np.linalg.solve(np.dot(tx.transpose(), tx), np.dot(tx.transpose(), y))
    MSE=compute_mse(y,tx,weight)
    return MSE,weight