# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""
import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    y=1./(1+np.exp(-t))
    return y

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

def compute_sig_loss(y,tx,w):
    """correct the output and compute the loss of logistic function"""
    y=(y+1)/2
    loss=np.log(1+np.exp(tx.dot(w)))
    loss=np.sum(loss)
    loss-=y.T.dot(tx).dot(w)
    return loss

def compute_sig_loss_var(y,tx,w):
    """correct the output and compute another loss of logistic function"""
    y=(y+1)/2
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)