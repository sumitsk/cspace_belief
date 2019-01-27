import numpy
import dist
import random


# computes query's neighbors weight and returns cpr
def predictor(query, neighbors, neighbors_ccr, neighbors_weight, method, param=None, inv_cov=None):
    delta = neighbors - query
    w = numpy.ones(len(neighbors))

    for i in range(len(neighbors)):
        if method == 'r':               # reciprocal (1/d) 
            w[i] = dist.reciprocal(delta[i], inv_cov, neighbors_weight[i])   
        
        elif method == 'g':             # gaussian (exp(-d**2))
            w[i] = dist.gaussian(delta[i], param, inv_cov, neighbors_weight[i]) 
        
        else:                           # 'ep' - epanechnikov ( 0.75 * (1-d**2/r**2) )
            w[i] = dist.epanechnikov(delta[i], param, inv_cov, neighbors_weight[i])
        
        cpr = numpy.dot(w, neighbors_ccr) / numpy.sum(w)

    # cpr  = bnb_predictor(query, neighbors, neighbors_ccr, w)    
    return cpr 

# Nearest Neighbors collision predictor
# NN -> nbrs , ANN -> lshf
def nn_predictor(query, training_set, training_set_ccr, nbrs, n_neighbors, method, weights=None, inv_cov=None):
    
    indices = neighbors_indices(query, nbrs, 'knn', n_neighbors)

    neighbors = training_set[indices]
    neighbors_ccr = training_set_ccr[indices]

    if inv_cov is None:
        inv_cov = numpy.identity(len(query))
    
    if weights is None:
        neighbors_weight = numpy.ones((len(neighbors), len(neighbors[0])))
    else:
        neighbors_weight = weights[indices]
    
    return predictor(query, neighbors, neighbors_ccr, neighbors_weight, method, inv_cov=inv_cov)
    
# gaussian kernel predictor    
def gaussian_predictor(query, training_set, training_set_ccr, nbrs, radius, ks=None, inv_cov=None, weights=None):
    
    indices = neighbors_indices(query, nbrs, 'rnn', radius)
    
    if len(indices) == 0:
        return random.uniform(-1,1)

    if ks is None:
        ks = 1

    neighbors = training_set[indices]
    neighbors_ccr = training_set_ccr[indices]

    if weights is None:
        neighbors_weight = numpy.ones((len(neighbors), len(neighbors[0])))
    else:
        neighbors_weight = weights[indices]
        return predictor(query, neighbors, neighbors_ccr, neighbors_weight, 'g', param=ks, inv_cov=inv_cov)
    
def epanechnikov_predictor(query, training_set, training_set_ccr, nbrs, radius, inv_cov=None, weights=None):
    indices = neighbors_indices(query, nbrs, 'rnn', radius)

    if len(indices) == 0:
        return random.uniform(-1,1)

    neighbors = training_set[indices]
    neighbors_ccr = training_set_ccr[indices]

    if weights is None:
        neighbors_weight = numpy.ones((len(neighbors), len(neighbors[0])))
    else:
        neighbors_weight = weights[indices]

    return predictor(query, neighbors, neighbors_ccr, neighbors_weight, 'ep', param=radius, inv_cov=inv_cov)
   
def neighbors_indices(query, nbrs, nn_method, param=1):
    
    # ANN or NN with param no. of NN    
    if nn_method == 'knn':         # k NN
        distances, indices = nbrs.kneighbors(query, n_neighbors=param)
  
    # NN within radius = param
    else:
        distances, indices = nbrs.radius_neighbors(query, radius=param)
        
    return indices[0]


# brock and burns predictor
def bnb_predictor(query, neighbors, nccr, w):
    # x - query
    # y - ccr
    mu_y = numpy.dot(nccr,w) / numpy.sum(w)
    mu_x = numpy.dot(neighbors.transpose(),w) / numpy.sum(w)
    
    delta_y = nccr - mu_y
    delta_x = neighbors - mu_x
    temp = numpy.sum(delta_x**2, axis=1)
    var_x = numpy.dot(temp, w) / numpy.sum(w)
    var_xy = numpy.dot(numpy.multiply(delta_y,w),delta_x) / numpy.sum(w)

    if var_x == 0:
        return mu_y 

    cpr = mu_y + numpy.dot(var_xy,query-mu_x) / var_x
    return cpr

# gaussian and epanechnikov predictors
def kernel_predictor(query, training_set, training_set_ccr, nbrs, indices, radius, weights=None, inv_cov=None):

    neighbors = training_set[indices]
    neighbors_ccr = training_set_ccr[indices]

    if weights is None:
        neighbors_weight = numpy.ones((len(neighbors), len(neighbors[0])))
    else:
        neighbors_weight = weights[indices]

    wg = numpy.ones(len(neighbors))
    wep = numpy.ones(len(neighbors))

    delta = neighbors - query
    
    ks = 1.0
    
    for i in range(len(neighbors)):
        wg[i] = dist.gaussian(delta[i], ks, inv_cov, neighbors_weight[i]) 
        wep[i] = dist.epanechnikov(delta[i], radius, inv_cov, neighbors_weight[i]) 
    
    wg = wg / numpy.linalg.norm(wg)
    wep = wep / numpy.linalg.norm(wep)

    cprg = numpy.dot(wg, neighbors_ccr) / numpy.sum(wg)
    cprep = numpy.dot(wep, neighbors_ccr) / numpy.sum(wep)

    return cprg, cprep
