# -*- coding: utf-8 -*-
"""

Split the dataset based on the given ratio.
"""


import numpy as np
from utils_functions import ridge_regression
from costs import compute_mse


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    size=x.shape[0]
    indices=np.random.permutation(size)
    ind_train=indices[:int(ratio*size)]
    ind_test=indices[int(ratio*size):]
    return x[ind_train],y[ind_train],x[ind_test],y[ind_test]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """train the model over the kth batch and test it on the rest of the given dataset."""
    test_ind=k_indices[k]
    train_ind=k_indices[[i for i in range(k_indices.shape[0]) if i!=k]]
    train_ind=train_ind.reshape(-1)
    
    y_train=y[train_ind]
    y_test=y[test_ind]
    x_train=x[train_ind,:]
    x_test=x[test_ind,:]

    w,loss_tr=ridge_regression(y_train,x_train,lambda_)

    loss_te=compute_mse(y_test,x_test,w)
    return loss_tr, loss_te

def reg_log_cross_validation(y, tx, k_indices, k, lambda_,gamma):
    """train the model over the kth batch and test it on the rest of the given dataset."""
    test_ind=k_indices[k]
    train_ind=k_indices[[i for i in range(k_indices.shape[0]) if i!=k]]
    train_ind=train_ind.reshape(-1)
    
    y_train=y[train_ind]
    y_test=y[test_ind]
    tx_train=tx[train_ind,:]
    tx_test=tx[test_ind,:]

    w,loss_tr=reg_logistic_regression(y_train,tx_train,lambda_,np.zeros(tx.shape[1]),100,gamma)
    loss_te=compute_sig_loss2(y_test,tx_test,w)+lambda_/2*np.linalg.norm(w)
    return loss_tr, loss_te

def cross_validation_SGD(y, x, k_indices, k, gamma):
    """return the loss of ridge regression."""
    test_ind=k_indices[k]
    train_ind=k_indices[[i for i in range(k_indices.shape[0]) if i!=k]]
    train_ind=train_ind.reshape(-1)
    
    y_train=y[train_ind]
    y_test=y[test_ind]
    x_train=x[train_ind,:]
    x_test=x[test_ind,:]

    #tx_train=build_poly(x_train,degree)
    #tx_test=build_poly(x_test,degree)

    w,loss_tr=least_squares_SGD(y_train,x_train,np.zeros((x_train.shape[1],)),500,gamma)

    loss_te=compute_mse(y_test,x_test,w)
    return loss_tr, loss_te