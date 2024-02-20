import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline
X_train, X_val, y_val = load_data()

#print("The first 5 elements of X_train are:\n", X_train[:5]) 

# A scatter plot of the data.
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 
plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()


# GRADED FUNCTION: estimate_gaussian
def estimate_gaussian(X): 
#X is data matrix 
    
    m, n = X.shape
    mu = 1 / m * np.sum(X, axis = 0)
    var = 1 / m * np.sum((X - mu) ** 2, axis = 0)
        
    return mu, var


# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)
    
# UNIT TEST
from public_tests import *
estimate_gaussian_test(estimate_gaussian)

# Returns the density of the multivariate normal
p = multivariate_gaussian(X_train, mu, var)

#Plot
visualize_fit(X_train, mu, var)

# UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val): 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    tp, fp, fn = 0, 0, 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        predictions = p_val < epsilon #if true then an anomaly

        #y_val (ndarray): Ground truth on validation set
        #p_val (ndarray): Results on validation set

        tp = ((predictions == True) & (y_val == True)).sum()
        fp = ((predictions == True) & (y_val == False)).sum()
        fn = ((predictions == False) & (y_val == True)).sum()

        #print(fp, tp, fn)
        
        prec = tp / (tp + fp)
        
        rec = tp / (tp + fn)
        
        F1 = ( 2 * prec * rec)/(prec + rec)
        
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)
    
# UNIT TEST
select_threshold_test(select_threshold)

# Find the outliers in the training set 
outliers = p < epsilon


visualize_fit(X_train, mu, var)

# Red circle around outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)