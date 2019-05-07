#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import os
import knn

import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    files = (['env_shelf01', 'env_table1', 'env_table3',
              'env_shelf02', 'env_kitchen1', 'env_kitchen2',
              'env_kitchen_refrigerator', 'env_kitchen_microwave'])  

    local_path = os.getcwd()
    training_set_path = os.path.join(local_path, "imp_samples/sobol_samples_1_7/")
    test_set_path = os.path.join(local_path, "test_set/")
    results_path = os.path.join(local_path, "method_results/")

    cfree_val = -1.0
    cobs_val = 1.0
    threshold = 0.0
    d = cobs_val - cfree_val
    dim = 7

    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    N_values = [1000, 5000, 10000, 15000, 20000]

    avg_accg = np.zeros((len(N_values), len(r_values)))
    avg_errg = np.zeros((len(N_values), len(r_values)))
    avg_accep = np.zeros((len(N_values), len(r_values)))
    avg_errep = np.zeros((len(N_values), len(r_values)))

    for i in range(len(files)):
        print('------------------', files[i], '----------------')
        accg = np.ones((len(N_values), len(r_values)))
        errg = np.ones((len(N_values), len(r_values)))
        accep = np.ones((len(N_values), len(r_values)))
        errep = np.ones((len(N_values), len(r_values)))
        
        for j in range(len(N_values)):
            training_set_size = N_values[j]
            test_set_size = training_set_size / 10
            
            print("training_set_size:", training_set_size)
            fn = 'sobol_' + files[i] + '_' + str(training_set_size) +'.npz'
            n = np.load(os.path.join(training_set_path,fn))
            training_set = n['samples']
            training_set_ccr = n['ccr']
            sjw = n['sjw']
            S = np.cov(training_set.transpose())
            inv_cov = np.linalg.inv(S)
            
            fn1 = files[i] + '_' + str(training_set_size) +'.npz'
            n1 = np.load(os.path.join(test_set_path,fn1))
            test_set = n1['test_set']
            ccr = n1['ccr']
                      
            nbrs = NearestNeighbors()
            nbrs.fit(training_set)
            
            cprg = np.ones((test_set_size, len(r_values)))
            cprep = np.ones((test_set_size, len(r_values)))

            idx = 0
            while idx < test_set_size:
                query = test_set[idx]
                for t in range(len(r_values)):  
                    r = r_values[t]
                    
                    cprg[idx,t] = knn.gaussian_predictor(query, training_set, training_set_ccr, nbrs, r,
                                                         ks=1.0, inv_cov=inv_cov, weights=sjw)
                    cprep[idx,t] = knn.epanechnikov_predictor(query, training_set, training_set_ccr, nbrs, r,
                                                              inv_cov=inv_cov, weights=sjw)

                idx = idx + 1
                
            for t in range(len(r_values)):  
                accg[j,t] = 1.0 * np.sum((cprg[:,t] > threshold) == (ccr == cobs_val)) / test_set_size
                accep[j,t] = 1.0 * np.sum((cprep[:,t] > threshold) == (ccr == cobs_val)) / test_set_size

                errg[j,t] = np.sum(np.absolute(cprg[:,t] - ccr)) / d / test_set_size
                errep[j,t] = np.sum(np.absolute(cprep[:,t] - ccr)) / d / test_set_size

                print(accg[j,t], errg[j,t], accep[j,t], errep[j,t])    

        avg_accg = avg_accg + accg
        avg_errg = avg_errg + errg
        avg_accep = avg_accep + accep
        avg_errep = avg_errep + errep

        # np.savez(os.path.join(results_path, files[i]+'_krnl_m_w'),
        #             accg=accg, accep=accep, errg=errg, errep=errep)

    avg_accg = avg_accg / len(files)
    avg_errg = avg_errg / len(files)
    avg_accep = avg_accep / len(files)
    avg_errep = avg_errep / len(files)
      
    print("average_values:")                
    print(avg_accg, avg_accep)
    print(avg_errg, avg_errep)
