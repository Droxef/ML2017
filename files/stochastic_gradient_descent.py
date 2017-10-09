# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-np.dot(tx,w)
    grad=-1/y.shape[0]*np.dot(np.transpose(tx),e)
    return grad,e


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for yBatch,txBatch in batch_iter(y, tx, batch_size):
            grad,_=compute_stoch_gradient(yBatch,txBatch,w)
            w=w-gamma*grad
            loss=compute_loss(y,tx,w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws