# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    #data=np.vstack((x,y))
    #np.random.shuffle(data.T)
    #data=data.T
    #size=x.shape[0]
    #train=data[:int(ratio*size),:]
    #test=data[int(ratio*size):,:]
    #return train[:,0],train[:,1],test[:,0],test[:,1]

    size=x.shape[0]
    indices=np.random.permutation(size)
    ind_train=indices[:int(ratio*size)]
    ind_test=indices[int(ratio*size):]
    return x[ind_train],y[ind_train],x[ind_test],y[ind_test]
