import numpy as np
import math

# method                - 'None' arguments
# L2                    - weights, inv_cov 
# weighted L2           - inv_cov 
# mahalanobis           - weights 
# weighted mahalonbis   - 

def weighted_L2(delta, inv_cov=None, weights=None):
    
    # inv_cov - covariance matrix inverse
    # weights - joint weights
    
    if weights is None:
        weights = np.ones(len(delta))
    delta = np.multiply(delta, weights**0.5)
        
    if inv_cov is None:
        inv_cov = np.identity(len(delta))    
                
    return math.sqrt(np.dot(delta, np.dot(inv_cov, delta)))


def reciprocal(delta, inv_cov=None, weights=None):
    return 1.0/weighted_L2(delta, inv_cov, weights)


def gaussian(delta, ks, inv_cov=None, weights=None):
    return math.exp(-ks * (weighted_L2(delta, inv_cov, weights) **2))


def epanechnikov(delta, radius, inv_cov=None, weights=None):
    return 0.75 * (1 - (weighted_L2(delta, inv_cov, weights))**2 / radius**2)
        
