#!/usr/bin/env python -W ignore::DeprecationWarning

import os
import numpy

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
    r_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    N_values = [1000, 5000, 10000, 15000, 20000]

    avg_acc = numpy.zeros(len(N_values)) # , len(r_values))) 
    avg_err = numpy.zeros(len(N_values)) # , len(r_values)))
    avg_accw = numpy.zeros(len(N_values)) # , len(r_values)))
    avg_errw = numpy.zeros(len(N_values)) # , len(r_values)))
    avg_accm = numpy.zeros(len(N_values))
    avg_errm = numpy.zeros(len(N_values))
    avg_accmw = numpy.zeros(len(N_values))
    avg_errmw = numpy.zeros(len(N_values))

    for i in range(len(files)):
        # print('------------------', files[i], '------------------'
     
        fn = files[i] + "_DT.npz"
        n = numpy.load(os.path.join(results_path, fn))
        accw = n['accw']
        acc = n['acc']
        errw = n['errw']
        err = n['err']
        accmw = n['accmw']
        accm = n['accm']
        errmw = n['errmw']
        errm = n['errm']
        
        
        avg_acc = avg_acc + acc
        avg_err = avg_err + err
        avg_accw = avg_accw + accw
        avg_errw = avg_errw + errw
        avg_accm = avg_accm + accm
        avg_errm = avg_errm + errm
        avg_accmw = avg_accmw + accmw
        avg_errmw = avg_errmw + errmw
        
    avg_acc = avg_acc / len(files)
    avg_accw = avg_accw / len(files)
    avg_err = avg_err / len(files)
    avg_errw = avg_errw / len(files)
    avg_accm = avg_accm / len(files)
    avg_accmw = avg_accmw / len(files)
    avg_errm = avg_errm / len(files)
    avg_errmw = avg_errmw / len(files)
    
    numpy.savez(os.path.join(results_path, 'avg_DT'),
                acc=avg_acc,
                accw=avg_accw,
                err=avg_err,
                errw=avg_errw,
                accm=avg_accm,
                accmw=avg_accmw,
                errm=avg_errm,
                errmw=avg_errmw) 
    
    print(avg_acc)
    print(avg_accm)
    print(avg_accw)
    print(avg_accmw)
    print(avg_err)
    print(avg_errm)
    print(avg_errw)
    print(avg_errmw)
