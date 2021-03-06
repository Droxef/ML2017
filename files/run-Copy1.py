# -*- coding: utf-8 -*-
"""

main program that train a model over a dataset and create a submission csv for kaggle
"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from helpers import *
from utils_functions import *
from split_data import *

if __name__=="__main__":
    max_iters=500
    lambda_=0
    degree=1
    seed=1
    k_fold=4
    # Load the data
    testY,testX,idTest=load_csv_data("../../dataset/test.csv", sub_sample=False)
    trainY,trainX,idTrain=load_csv_data("../../dataset/train.csv", sub_sample=False)
    # replace unkown data and replace it by the mean of the column
    testX=clean_data(testX)
    trainX=clean_data(trainX)
    # transform skewed features in log 
    for i in range(trainX.shape[1]):
        if(not np.any(trainX[:,i]<=0)):
            trainX[:,i]=np.log(trainX[:,i])
        else:
            trainX[:,i]=np.log(trainX[:,i]-np.min(trainX[:,i])+1)
    for i in range(testX.shape[1]):
        if(not np.any(testX[:,i]<=0)):
            testX[:,i]=np.log(testX[:,i])
        else:
            testX[:,i]=np.log(testX[:,i]-np.min(testX[:,i])+1)
    #standardize
    testX=standardize(testX)
    trainX=standardize(trainX)
    #reject outliers
    testX=reject_outliers(testX)
    trainX=reject_outliers(trainX)
    
    ### check best degree with cross val
    degrees=range(1,15)
    lambda_=0
    k_indices = build_k_indices(trainY, k_fold, seed)


    losses=[]
    losses_te=[]
    stds=[]

    for degree in degrees:
        tx = build_poly(trainX,degree)
        temp_tr=[]
        temp_te=[]
        for k in range(k_fold):
            tr,te=cross_validation(trainY,tx,k_indices,k,lambda_)
            temp_tr.append(tr)
            temp_te.append(te)
        losses.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
        stds.append(np.std(temp_te))
    del losses_te[np.argmin(losses_te)]
    degree=degrees[np.argmin(losses_te)]
    
    ### check best lambda with cross val
    lambdas= np.logspace(-8, 0, 50)

    losses=[]
    losses_te=[]
    stds=[]
    tx = build_poly(trainX,degree)

    for lambda_ in lambdas:
        temp_tr=[]
        temp_te=[]
        for k in range(k_fold):
            tr,te=cross_validation(trainY,tx,k_indices,k,lambda_)
            temp_tr.append(tr)
            temp_te.append(te)
        losses.append(np.mean(temp_tr))
        losses_te.append(np.mean(temp_te))
        stds.append(np.std(temp_te))
    lambda_=lambdas[np.argmin(losses_te)]
    
    # Find weights with best hyper-parameters
    w,loss=ridge_regression(trainY,tx,lambda_)
    
    # Final submission for test set
    tx_test=build_poly_full(testX,degree)
    ypred=predict_labels(w,tx_test)
    create_csv_submission(idTest,ypred,"submission.csv")
