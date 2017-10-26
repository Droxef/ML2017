# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    Xmat=np.ones(x.shape)
    for i in range(1,degree+1):
        Xmat=np.hstack([Xmat,np.power(x,i)])
    return Xmat

def build_cross_poly(tx):
    """add cross influence of second order. Higher order will be estimated as null"""
    poly = np.ones((tx.shape[0], 1))
    for feat in range(tx.shape[1]):
        for sec in range(feat+1,tx.shape[1]):
            poly = np.c_[poly, tx[:,feat]*tx[:,sec]]
    return poly

def build_poly_full(tx,degree,cross=False):
    """polynomial basis function up to degree and adding cross interaction of second order if more than one feature"""
    Xmat=build_poly(tx,degree)
    if cross:
        Xmat=np.c_[Xmat, build_cross_poly(tx)[:,1:]]
    return Xmat