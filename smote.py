from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class 
             observations in the output
    k - k in k nearest neighbors

    OUTPUT:
    X_oversampled, y_oversampled - oversampled data

    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    
    y_zeros = y[y==0]
    X_zeros = X[y==0]
    
    y_ones = y[y==1]
    X_ones = X[y==1]
    
    if len(y_ones) > len(y_zeros):
        y_minority = y_zeros
        X_minority = X_zeros
    else:
        y_minority = y_ones
        X_minority = X_ones
    
    # fit a KNN model    
    # This has to be called on the minority bunch only!!!!!    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_minority)
    distances, indices = nbrs.kneighbors(X_minority)

    # determine how many new positive observations to generate    
    target = float(target)    
    N_new_data = (len(y)*target - len(y_minority))/(1-target)

    # adding to the zeros
    ind_new = np.random.randint(0,len(y_minority),N_new_data)

    
    # generate synthetic observations
    
    y_synth = np.zeros(len(N_new_data))
    X_synth = []        
    for value in ind_new:
        r = np.random(0, k)           
        neighbor_index = indices[value, r]
        distances = np.random.random(0, 1, len(X_minority.columns))            
        new_point = X_minority[value] + distances*(X_minority[value]-X_minority[neighbor_index])            
        X_synth.append = new_point

    # combine synthetic observations with original observations
    X_smoted = np.concatenate((X_ones, X_zeros, X_synth),axis=1)
    y_smoted = np.concatenate((y_ones, y_zeros, y_synth),axis=1)
   
    return X_smoted, y_smoted
