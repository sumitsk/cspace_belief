#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy as np

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

    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    N_values = [1000, 5000, 10000, 15000, 20000]

    avg_accg = np.zeros((len(N_values), len(r_values)))  
    avg_errg = np.zeros((len(N_values), len(r_values))) 
    avg_accep = np.zeros((len(N_values), len(r_values))) 
    avg_errep = np.zeros((len(N_values), len(r_values)))
    '''
    avg_accm = np.zeros(  len(N_values) )
    avg_errm = np.zeros(  len(N_values) )
    avg_accmw = np.zeros(  len(N_values) )
    avg_errmw = np.zeros(  len(N_values) )
    '''
    for i in range(len(files)):
        # print('------------------', files[i], '------------------'
     
        fn = files[i] + "_krnl_m.npz"
        n = np.load(os.path.join(results_path, fn))
        accep = n['accep']
        accg = n['accg']
        errep = n['errep']
        errg = n['errg']
        '''
        accmw = n['accmw']
        accm = n['accm']
        errmw = n['errmw']
        errm = n['errm']
        '''
        
        avg_accg = avg_accg + accg
        avg_errg = avg_errg + errg
        avg_accep = avg_accep + accep
        avg_errep = avg_errep + errep
        '''
        avg_accm = avg_accm + accm
        avg_errm = avg_errm + errm
        avg_accmw = avg_accmw + accmw
        avg_errmw = avg_errmw + errmw
        '''
    avg_accg = avg_accg / len(files)
    avg_accep = avg_accep / len(files)
    avg_errg = avg_errg / len(files)
    avg_errep = avg_errep / len(files)
    '''
    avg_accm = avg_accm / len(files)
    avg_accmw = avg_accmw / len(files)
    avg_errm = avg_errm / len(files)
    avg_errmw = avg_errmw / len(files)
    '''
    np.savez(os.path.join(results_path, 'avg_krnl_m'),
             accg=avg_accg,
             accep=avg_accep,
             errg=avg_errg,
             errep=avg_errep)
    
    print(avg_accg)
    print(avg_accep)
    print(avg_errg)
    print(avg_errep)
    # print(avg_errm)
    # print(avg_errmw)
    # print(avg_accm)
    # print(avg_accmw)
