# -*- coding: utf-8 -*-
"""

All method functions required for project 1
"""


import numpy as np
from costs import *
from helpers import *
from gradient_descent import *
from build_polynomial import *

def least_squares_GD(y,tx,initial_w,max_iters,gamma):
    """ gradient descent using least squares method """
    threshold = 1e-8
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y,tx,w)
        old_loss=compute_mse(y,tx,w)
        w=w-gamma*grad
        loss=compute_mse(y,tx,w)
        if (old_loss-loss)<threshold:
            break
    return w, loss

def least_squares_SGD(y,tx,initial_w,max_iters,gamma):
    """ stochastic gradient descent using least squares method """
    threshold = 1e-8
    w = initial_w
    batch_size=1 # mandatory batch size of 1
    for n_iter in range(max_iters):
        for yBatch,txBatch in batch_iter(y, tx, batch_size):
            grad=compute_gradient(yBatch,txBatch,w)
            old_loss=compute_mse(y,tx,w)
            w=w-gamma*grad
            loss=compute_mse(y,tx,w)
            if (old_loss-loss)<threshold:
                break
    return w, loss

def least_squares(y,tx):
    """ find optimal weights using least squares """
    if len(y)!=tx.shape[0]:
        raise ValueError("y and tx must have the same length")
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss=compute_mse(y,tx,w)
    return w, loss

def ridge_regression(y,tx,lambda_):
    """ adding regularization to avoid too complicated model, with least squares regression """
    w=np.linalg.solve(tx.T.dot(tx)+2*tx.shape[0]*lambda_*np.eye(tx.shape[1]),tx.T.dot(y))
    loss=compute_mse(y,tx,w)
    return w, loss

def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """ gradient descent using logistic loss function """
    threshold = 1e-8
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_sig_gradient(y,tx,w)
        old_loss=compute_sig_loss(y,tx,w)
        w-=gamma*grad
        loss=compute_sig_loss(y,tx,w)
        if (old_loss-loss)<threshold:
            break
        #if n_iter%100==0:
        #    print(loss)
    return w, loss

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """ gradient descent using logistic function with regularization """
    threshold = 1e-8
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_sig_gradient(y,tx,w)+2*lambda_*w
        old_loss=compute_sig_loss(y,tx,w)+lambda_/2*np.linalg.norm(w)
        w-=gamma*grad
        loss=compute_sig_loss(y,tx,w)+lambda_/2*np.linalg.norm(w)
        if (old_loss-loss)<threshold:
            break
    return w, loss
