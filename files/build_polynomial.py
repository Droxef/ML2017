# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    Xmat=np.ones(x.shape)
    for i in range(1,degree+1):
        Xmat=np.vstack([Xmat,np.power(x,i)])
    return Xmat.transpose()
    
    #Xmat=np.ones(x.shape[0],degree)
    #for i in range(degree):
    #    Xmat[:,i]=np.power(x,i)
    #return Xmat