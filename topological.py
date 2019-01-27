import numpy
import os
import random
from scipy.spatial import Delaunay

import knn

def find_cprs(query, tri, ts1, ts2, training_set_ccr, inv_cov1=None, inv_cov2=None, weights=None):

    dim = len(ts1[0])
    q1 = query[:dim] 
    si_idx = tri.find_simplex(q1)

    if si_idx == -1:
        cpr = random.uniform(-1,1)
        cprw = random.uniform(-1,1)
        return cpr, cprw
        
    neighbors_idx = tri.simplices[si_idx]

    neighbors1 = ts1[neighbors_idx]
    neighbors2 = ts2[neighbors_idx]

    if sjw is None:
        neighbors_weight1 = numpy.ones((len(neighbors1), len(neighbors1[0])))
        neighbors_weight2 = numpy.ones((len(neighbors2), len(neighbors2[0])))
    else:
        neighbors_weight1 = sjw[neighbors_idx,:dim]
        neighbors_weight2 = sjw[neighbors_idx,dim:]

    nccr = training_set_ccr[neighbors_idx]
    q1 = query[:dim]
    q2 = query[dim:]

    cpr1 = knn.predictor(q1, neighbors1, nccr, neighbors_weight1, 'r', inv_cov=inv_cov1)
    cpr2=knn.predictor(q2, neighbors2, nccr, neighbors_weight2, 'r', inv_cov=inv_cov2)
    return cpr1, cpr2


def find_best_alpha(training_set, sjw=None):
    
    training_set_size = len(training_set)   
    val_set_size = training_set_size / 5 
    idx1 = random.sample(range(0, training_set_size/2), val_set_size/2)
    idx2 = random.sample(range(training_set_size/2, training_set_size), val_set_size/2)

    idxv = idx1 + idx2
    val_set = training_set[idxv]
    val_ccr = training_set_ccr[idxv]

    idxt = list(set(range(0,training_set_size)) - set(idxv))
    trn_set = training_set[idxt]
    trn_ccr = training_set_ccr[idxt]

    if sjw is None:
        trn_jw = numpy.ones((len(trn_set), len(trn_set[0])))
    else:
        trn_jw = sjw[idxt]

    ts1 = trn_set[:,:4]
    ts2 = trn_set[:,4:]
    tri = Delaunay(ts1)

    # s - covariance matrix
    S1 = numpy.cov(ts1.transpose())
    S2 = numpy.cov(ts2.transpose())

    threshold = 0.0
    max_acc = 0.0
    min_err = 1.0

    cpr = numpy.ones(val_set_size)
    
    alpha = 0.01
    delta_alpha = 0.01
    best_alpha = alpha    
        
    while alpha < 1.0:
        i = 0
        while i < val_set_size:  
            si_idx = tri.find_simplex(val_set[i][:4])
            k = int(si_idx)
            if k == -1:
                cpr[i] = numpy.uniform(-1,1)
                i = i + 1
                continue      

            cpr1, cpr2 = find_cprs(val_set[i], tri, ts1, ts2, trn_ccr, inv_cov1, inv_cov2, trn_jw)
            cpr[i] = (cpr1 + alpha*cpr2)/(1+alpha)
            i = i + 1

        # print(cpr
        # acc = 1.0 * numpy.sum((val_ccr == 1) == (cpr > threshold)) / val_set_size
        err = numpy.sum(numpy.absolute(val_ccr-cpr)) / val_set_size / d
        
        if err > min_err:
            min_err = err
            best_alpha = alpha
        alpha = alpha + delta_alpha

        # print(a
        # raw_input("sdf{")
    return best_alpha

if __name__ == '__main__':
    files = (['env_table1', 'env_table3', 'env_shelf01',
              'env_shelf02', 'env_kitchen1', 'env_kitchen2',
              'env_kitchen_refrigerator', 'env_kitchen_microwave'])  
    
    local_path = os.getcwd()
    training_set_path = os.path.join(local_path, "imp_samples/sobol_samples_1_7/")
    test_set_path = os.path.join(local_path, "test_set/")
    results_path = os.path.join(local_path, "method_results/")

    dim = 7
    cfree_val = -1.0
    cobs_val = 1.0
    d = cobs_val - cfree_val
    threshold = (cfree_val + cobs_val)/2
    
    N_values = [1000, 5000, 10000, 15000, 20000]

    for i in range(len(files)):
        print('--------------------',files[i],'------------------')
        
        acc = numpy.zeros(len(N_values))
        accw = numpy.zeros(len(N_values))
        accm = numpy.zeros(len(N_values))
        accmw = numpy.zeros(len(N_values))
        err = numpy.zeros(len(N_values))
        errw = numpy.zeros(len(N_values))
        errm = numpy.zeros(len(N_values))
        errmw = numpy.zeros(len(N_values))
        
        for j in range(len(N_values)):
            training_set_size = N_values[j]
            test_set_size = training_set_size / 10

            # print("training_set_size:", training_set_size
            fn = 'sobol_' + files[i] + '_' + str(training_set_size) + '.npz'
            n = numpy.load(os.path.join(training_set_path,fn))
            training_set = n['samples']
            training_set_ccr = n['ccr']
            sjw = n['sjw']
            
            fn1 = files[i] + '_' + str(N_values[j]) + '.npz'
            n1 = numpy.load(os.path.join(test_set_path, fn1))
            test_set = n1['test_set']
            ccr = n1['ccr']

            ts1 = training_set[:,:4]
            ts2 = training_set[:,4:]

            tri = Delaunay(ts1)

            # s - covariance matrix
            S1 = numpy.cov(ts1.transpose())
            S2 = numpy.cov(ts2.transpose())
            inv_cov1 = numpy.linalg.inv(S1)
            inv_cov2 = numpy.linalg.inv(S2)
            
            cpr = numpy.ones(test_set_size)
            cprw = numpy.ones(test_set_size)
            cprm = numpy.ones(test_set_size)
            cprmw = numpy.ones(test_set_size)
            '''
            cpr1 = numpy.ones(test_set_size)
            cpr2 = numpy.ones(test_set_size)
            cprw1 = numpy.ones(test_set_size)
            cprw2 = numpy.ones(test_set_size)
            '''
            
            # hard coded value            
            alpha = 0.1        
            
            idx = 0
            while idx < test_set_size:
                query = test_set[idx]
                                 
                cpr1, cpr2 = find_cprs(query, tri, ts1, ts2, training_set_ccr)
                cpr[idx] = (cpr1 + alpha*cpr2)/(1+alpha)
                cpr1, cpr2 = find_cprs(query, tri, ts1, ts2, training_set_ccr, weights=sjw)
                cprw[idx] = (cpr1 + alpha*cpr2)/(1+alpha)
                cpr1, cpr2 = find_cprs(query, tri, ts1, ts2, training_set_ccr, inv_cov1=inv_cov1, inv_cov2=inv_cov2)
                cprm[idx] = (cpr1 + alpha*cpr2)/(1+alpha)
                cpr1, cpr2 = find_cprs(query, tri, ts1, ts2, training_set_ccr, inv_cov1=inv_cov1, inv_cov2=inv_cov2,
                                       weights=sjw)
                cprmw[idx] = (cpr1 + alpha*cpr2)/(1+alpha)
                
                # cpr1[idx], cpr2[idx] = find_cprs(query, tri, ts1, ts2, training_set_ccr, S1, S2)
                # cprw1[idx], cprw2[idx] = find_cprs(query, tri, ts1, ts2, training_set_ccr, S1, S2)
                idx = idx + 1
                
            acc[j] = 1.0 * numpy.sum((ccr == cobs_val) == (cpr > threshold)) / test_set_size
            err[j] = numpy.sum(numpy.absolute(ccr-cpr)) / test_set_size / d
            
            accw[j] = 1.0 * numpy.sum((ccr == cobs_val) == (cprw > threshold)) / test_set_size
            errw[j] = numpy.sum(numpy.absolute(ccr-cprw)) / test_set_size / d
            
            accm[j] = 1.0 * numpy.sum((ccr == cobs_val) == (cprm > threshold)) / test_set_size
            errm[j] = numpy.sum(numpy.absolute(ccr-cprm)) / test_set_size / d
            
            accmw[j] = 1.0 * numpy.sum((ccr == cobs_val) == (cprmw > threshold)) / test_set_size
            errmw[j] = numpy.sum(numpy.absolute(ccr-cprmw)) / test_set_size / d
                
        # numpy.savez( os.path.join(results_path, files[i]+'_DT') , acc = acc, accw = accw, accm = accm, accmw = accmw, err = err, errw = errw, errm = errm, errmw = errmw)
            
        print(acc,'\n', accw,'\n', accm,'\n', accmw)
        print(err,'\n', errw,'\n', errm,'\n', errmw)                

        '''
        accl = []
        accwl = []
        errl = []
        errwl = []
        x = []
        
        while alpha <= 1.0:
            cpr = (cpr1 + alpha * cpr2)/(1 + alpha)
            cprw = (cprw1 + alpha * cprw2)/(1 + alpha)
            
            acc = 1.0 * numpy.sum((ccr == cobs_val) == (cpr > threshold)) / test_set_size
            err = numpy.sum(numpy.absolute(ccr-cpr)) / test_set_size / d
            accw = 1.0 * numpy.sum((ccr == cobs_val) == (cprw > threshold)) / test_set_size
            errw = numpy.sum(numpy.absolute(ccr-cprw)) / test_set_size / d

            accl.append(acc)
            accwl.append(accw)
            errl.append(err)
            errwl.append(errw)                
            x.append(alpha)
            
            alpha = alpha + 0.01
            
        plt.figure(1)    
        plt.scatter(x,accl)
        plt.figure(2)    
        plt.scatter(x,accwl)
        plt.show()
        #print(accl, accwl, errl, errwl                
        '''            
