# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def clean_data(x, col=None):
    """ Remove -999 values and replace by mean of feature.
        May remove col features.
    """
    if col is not None:
        x=np.delete(x,col,1)
    x[x==-999]=np.nan
    meanX=np.nanmean(x,axis=0)
    indsX = np.where(np.isnan(x))
    x[indsX]=np.take(meanX,indsX[1])
    return x
    
def transform_data(x):
    """
    Transform the data in a log way if possible
    (and if data is skewed)
    """
    for i in range(x.shape[1]):
        if(not np.any(x[:,i]<=0)):
            x[:,i]=np.log(x[:,i])
    return x

def reject_outliers(data, m=10):
    """reject outliers if outside m*std"""
    newData=np.copy(data)
    ind=abs(data - np.mean(data,axis=0)) > m * np.std(data,axis=0)
    newData[ind]=0
    return newData
    

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
