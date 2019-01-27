#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy
import os
import knn

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

    avg_accnn = numpy.zeros((len(N_values), len(k_values))) 
    avg_errnn = numpy.zeros((len(N_values), len(k_values))) 
    avg_accann = numpy.zeros((len(N_values), len(k_values))) 
    avg_errann = numpy.zeros((len(N_values), len(k_values))) 

    for i in range(len(files)):
        print('------------------', files[i], '------------------')
        accnn = numpy.zeros((len(N_values), len(k_values))) 
        errnn = numpy.zeros((len(N_values), len(k_values))) 
        accann = numpy.zeros((len(N_values), len(k_values))) 
        errann = numpy.zeros((len(N_values), len(k_values))) 
        accmnn = numpy.zeros((len(N_values), len(k_values))) 
        errmnn = numpy.zeros((len(N_values), len(k_values))) 
        accmann = numpy.zeros((len(N_values), len(k_values))) 
        errmann = numpy.zeros((len(N_values), len(k_values))) 
        
        for j in range(len(N_values)):
            training_set_size = N_values[j]
            test_set_size = training_set_size / 10
            
            print("training_set_size:", training_set_size)
            fn = 'sobol_' + files[i] + '_' + str(training_set_size) + '.npz'
            n = numpy.load(os.path.join(training_set_path,fn))
            training_set = n['samples']
            training_set_ccr = n['ccr']
            sjw = n['sjw']
            S = numpy.cov(training_set.transpose())
            inv_cov = numpy.linalg.inv(S)
            
            fn1 = files[i] + '_' + str(N_values[j]) + '.npz'
            n1 = numpy.load(os.path.join(test_set_path, fn1))
            test_set = n1['test_set']
            ccr = n1['ccr']

            lshf = LSHForest()
            lshf.fit(training_set)

            nbrs = NearestNeighbors()
            nbrs.fit(training_set)
            
            cprnn = numpy.ones((test_set_size, len(k_values)))
            cprann = numpy.ones((test_set_size, len(k_values)))
            cprmnn = numpy.ones((test_set_size, len(k_values)))
            cprmann = numpy.ones((test_set_size, len(k_values)))

            idx = 0
            while idx < test_set_size:
                query = test_set[idx]

                for t in range(len(k_values)):  
                    n_neighbors = k_values[t]

                    cprnn[idx,t] = knn.nn_predictor(query, training_set, training_set_ccr, nbrs, n_neighbors,
                                                    method='r', weights=sjw)
                    cprann[idx,t] = knn.nn_predictor(query, training_set, training_set_ccr, lshf, n_neighbors,
                                                     method='r', weights=sjw)
                    cprmnn[idx,t] = knn.nn_predictor(query, training_set, training_set_ccr, nbrs, n_neighbors,
                                                     method='r', weights=sjw, inv_cov=inv_cov)
                    cprmann[idx,t] = knn.nn_predictor(query, training_set, training_set_ccr, lshf, n_neighbors,
                                                      method='r', weights=sjw, inv_cov=inv_cov)
                    # print(cprmann[idx,t] , cprann[idx,t]
                    # raw_input("s")    
                idx = idx + 1
        
            for t in range(len(k_values)):  
                accnn[j,t] = 1.0*numpy.sum((cprnn[:,t] > threshold) == (ccr == cobs_val)) / test_set_size
                accann[j,t] = 1.0*numpy.sum((cprann[:,t] > threshold) == (ccr == cobs_val)) / test_set_size

                errnn[j,t] = numpy.sum(numpy.absolute(cprnn[:,t] - ccr)) / d / test_set_size
                errann[j,t] = numpy.sum(numpy.absolute(cprann[:,t] - ccr)) / d / test_set_size

                accmnn[j,t] = 1.0*numpy.sum((cprmnn[:,t] > threshold) == (ccr == cobs_val)) / test_set_size
                accmann[j,t] = 1.0*numpy.sum((cprmann[:,t] > threshold) == (ccr == cobs_val)) / test_set_size

                errmnn[j,t] = numpy.sum(numpy.absolute(cprmnn[:,t] - ccr)) / d / test_set_size
                errmann[j,t] = numpy.sum(numpy.absolute(cprmann[:,t] - ccr)) / d / test_set_size

                # print(accnn[j,t] , errnn[j,t], accann[j,t] , errann[j,t]
                # print(accmnn[j,t] , errmnn[j,t], accmann[j,t] , errmann[j,t]    

        avg_accnn = avg_accnn + accnn
        avg_errnn = avg_errnn + errnn
        avg_accann = avg_accann + accann
        avg_errann = avg_errann + errann

        numpy.savez(os.path.join(results_path, files[i]+'_NN_w'),
                    accnn=accnn, accann=accann, errnn=errnn, errann=errann) 
        numpy.savez(os.path.join(results_path, files[i]+'_NN_m_w'),
                    accnn=accmnn, accann=accmann, errnn=errmnn, errann=errmann)
        print(accnn, accmnn)
        
    avg_accnn = avg_accnn / len(files)
    avg_errnn = avg_errnn / len(files)
    avg_accann = avg_accann / len(files)
    avg_errann = avg_errann / len(files)

    print("average_values:", avg_accnn, avg_accann, avg_errnn, avg_errann)
