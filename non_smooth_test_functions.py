# Define the function
import numpy as np
# defined between 0 and 1 for each dimension. 
def cosineMixture(X):
    X = np.array(X)
    z = -0.1*np.sum(np.cos(5*np.pi*X)) - np.sum(X**2)
    step = np.where(z>-0.4, 0.3, 0)
    z = z+step
    return z 
