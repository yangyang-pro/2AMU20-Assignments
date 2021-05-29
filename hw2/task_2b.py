from numpy.random import triangular
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv

class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        self.data = data
        self.root = root
        self.alpha = alpha 
        Tcsr = minimum_spanning_tree(data)
        Tcsr.toarray().astype(int)

    def get_tree(self):
        pass

    def get_log_params(self):
        pass

    def log_prob(self, x, exhaustive=False):
        pass

    def sample(self, n_samples):
        pass



if __name__ == "__main__":

    with open('baudio.train.data', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        train = np.array(list(reader)).astype(np.float)
        

    with open('baudio.test.data', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        test = np.array(list(reader)).astype(np.float)
        
    with open('baudio.valid.data', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        valid = np.array(list(reader)).astype(np.float)

    #run = BinaryCLT(train)
    print(train.shape)
   
  
