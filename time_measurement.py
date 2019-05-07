#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import os
import knn
import time

import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import NearestNeighbors, LSHForest

if __name__ == '__main__':
    files = (['env_shelf01', 'env_table1', 'env_table3',
              'env_shelf02', 'env_kitchen1', 'env_kitchen2',
              'env_kitchen_refrigerator', 'env_kitchen_microwave'])  
    
    local_path = os.getcwd()
    training_set_path = os.path.join(local_path, "imp_samples/sobol_samples_1_7/")
    test_set_path = os.path.join(local_path, "test_set/")
    results_path = os.path.join(local_path, "metric_results/")

    cfree_val = -1.0
    cobs_val = 1.0
    threshold = 0.0
    d = cobs_val - cfree_val
    dim = 7

    k_values = [1, 5, 10, 15, 20]
    N_values = [1000, 5000, 10000, 15000, 20000]
    
    avg_accnn = 0
    avg_errnn = 0
    avg_accann = 0
    avg_errann = 0
    
    tnn = 0
    tann = 0
    for i in range(len(files)):
        print('------------------', files[i], '------------------')
        accnn = np.zeros((len(N_values), 1)) 
        errnn = np.zeros((len(N_values), 1)) 
        accann = np.zeros((len(N_values), 1)) 
        errann = np.zeros((len(N_values), 1)) 
                
        N_value = 20000
        training_set_size = N_value
        test_set_size = training_set_size / 10
        
        fn = 'sobol_' + files[i] + '_' + str(training_set_size) + '.npz'
        n = np.load(os.path.join(training_set_path,fn))
        training_set = n['samples']
        training_set_ccr = n['ccr']
        
        fn1 = files[i] + '_' + str(N_value) + '.npz'
        n1 = np.load(os.path.join(test_set_path, fn1))
        test_set = n1['test_set']
        ccr = n1['ccr']

        t0 = time.clock()
        lshf = LSHForest()
        lshf.fit(training_set)
        tann = tann + time.clock() - t0

        t0 = time.clock()
        nbrs = NearestNeighbors()
        nbrs.fit(training_set)
        tnn = tnn + time.clock() - t0
        
        cprnn = np.ones((test_set_size, 1))
        cprann = np.ones((test_set_size, 1))

        idx = 0
        while idx < test_set_size:
            query = test_set[idx]

            n_neighbors = 10
            
            t0 = time.clock()
            cprnn[idx,0]=knn.nn_predictor(query, training_set, training_set_ccr, nbrs, n_neighbors,
                                            method='r', weights=None)
            tnn=tnn + time.clock() - t0
            
            t0=time.clock()
            cprann[idx,0]=knn.nn_predictor(query, training_set, training_set_ccr, lshf, n_neighbors,
                                             method='r', weights=None)
            tann = tann + time.clock() - t0
            
            idx = idx + 1
    
        t0 = time.clock()
        accnn = 1.0*np.sum((cprnn[:,0] > threshold) == (ccr == cobs_val)) / test_set_size
        errnn = np.sum(np.absolute(cprnn[:,0] - ccr)) / d / test_set_size
        tnn = tnn + time.clock() - t0
        
        t0 = time.clock()
        accann = 1.0*np.sum((cprann[:,0] > threshold) == (ccr == cobs_val)) / test_set_size
        errann = np.sum(np.absolute(cprann[:,0] - ccr)) / d / test_set_size
        tann = tann + time.clock() - t0
        
        t0 = time.clock()
        avg_accnn = avg_accnn + accnn
        avg_errnn = avg_errnn + errnn
        tnn = tnn + time.clock() - t0
        
        t0 = time.clock()
        avg_accann = avg_accann + accann
        avg_errann = avg_errann + errann
        tann = tann + time.clock() - t0
        
    t0 = time.clock()            
    avg_accnn = avg_accnn / len(files)
    avg_errnn = avg_errnn / len(files)
    tnn = tnn + time.clock() - t0
    
    t0 = time.clock()        
    avg_accann = avg_accann / len(files)
    avg_errann = avg_errann / len(files)
    tann = tann + time.clock() - t0

    print("time taken:", tnn, tann)                
    
