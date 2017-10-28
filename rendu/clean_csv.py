# -*- coding: utf-8 -*-
"""clean a dataset in csv file."""

import csv
import numpy as np

#replace before with provided value in array
def replace(array, before, value):
    
    f = lambda x,y: y if x==before else x
    if (np.isnan(before)):
        f = lambda x,y: y if np.isnan(x) else x
    vf = np.vectorize(f)
    #return np.fromiter((f(x, value) for x in array), array.dtype, count=len(array))
    return vf(array, value)

def mapp(array, f):
    return np.fromiter((f(x, value) for x in array), array.dtype, count=len(array))
        
def avg(array):
    return np.nanmean(array)

def create_f64_arrays(csv, size):
    header = next(csv)
    
    index = np.empty(size, dtype = np.int64)
    letters = np.empty(size, dtype = np.string_)
    ret = np.empty([size, len(header)-2], dtype = np.float64)

    
    idx = 0
    for x in csv:
        index[idx] = x[0]
        letters[idx] = x[1]
        
        ret[idx] = np.float64(x[2:])
        idx = idx + 1
    
    #print(ret)
    return (header, index, letters, ret)
    
    
def to_f64_try(input):
    try:
        ret =np.float64(input)

        return ret
    except ValueError:
        return input
    
def write_new_csv(name, header, index, letters, data):
    data_str = np.array(data, dtype='<U9')
    index_str = np.array(index, dtype='<U9')
    left = np.c_[index_str, letters]
    payload = np.c_[left, data_str]

    
    fp = open(name, 'w')
    wr = csv.writer(fp, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    wr.writerow(header)
    wr.writerows(payload)
    
    
    

    
if __name__=="__main__":
    bad_val = np.float64(-9.99000000e+02)
    zero = np.float64(0)

    # ------------------------------------------------
    #change these variables if necessary
    train_path = '../dataset/train.csv' # use argv maybe
    test_path = '../dataset/test.csv'

    train_cleaned_path = '../dataset/train_cleaned.csv'
    test_cleaned_path = '../dataset/test_cleaned.csv'

    #columns whose mean is below this number (absolute) will not be normalised (avoids numerical problems)
    normalisation_threshold = 0.01
    # ------------------------------------------------
    test = csv.reader(open(test_path, 'r'), delimiter=',')

    train = csv.reader(open(train_path, 'r'), delimiter=',')
    
    (test_header, test_index, test_letters, test_data) = create_f64_arrays(test, 568239-1)
    (train_header, train_index, train_letters, train_data) = create_f64_arrays(train, 250001-1)
    test_data_nan = replace(test_data, bad_val, np.nan)
    train_data_nan = replace(train_data, bad_val, np.nan)
    #replace -999 with column average
    for x in range(0, test_data_nan.shape[1]):
        aver = avg(test_data_nan[:, x])
        test_data_nan[:, x] = replace(test_data_nan[:, x], np.nan, aver)
        print(test_data_nan[:, x])

    for x in range(0, train_data_nan.shape[1]):
        aver = avg(train_data_nan[:, x])
        train_data_nan[:, x] = replace(train_data_nan[:, x], np.nan, aver)

    write_new_csv(test_cleaned_path, test_header, test_index, test_letters, test_data_norm)
    write_new_csv(train_cleaned_path, train_header, train_index, train_letters, train_data_norm)
