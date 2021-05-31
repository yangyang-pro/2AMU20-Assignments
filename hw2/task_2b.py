from numpy.random import triangular
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
import itertools


class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        self.data = data
        self.root = root
        self.alpha = alpha
        # for binary discrete random variables, the number of states is 2
        self.num_states = 2

    def compute_mi(self, marginals_x, marginals_y, joints_xy):
        mi = 0
        for val_x in range(self.num_states):
            for val_y in range(self.num_states):
                mi += joints_xy(val_x, val_y) * np.log(joints_xy[val_x, val_y] /
                                                       (marginals_x[val_x] * marginals_y[val_y]))
        return mi

    def create_clt(self):
        # number of samples according to the real distribution
        num_samples = self.data.shape[0]
        # number of random variables
        num_rvs = self.data.shape[1]

        rvs = np.arange(num_rvs)
        rv_combinations = list(itertools.combinations(rvs, 2))
        mi_matrix = np.zeros((num_rvs, num_rvs))
        for rv_combination in rv_combinations:
            # current random variables
            x, y = rv_combination
            # samples for current random variables
            samples_xy = self.data[:, rv_combination]
            num_samples = samples_xy.shape[0]
            samples_x = samples_xy[:, 0]
            samples_y = samples_xy[:, 1]

            marginal_probs_x = np.zeros(self.num_states)
            marginal_probs_y = np.zeros(self.num_states)
            for val in range(self.num_states):
                num_samples_x, num_samples_y = sum(samples_x == val), sum(samples_y == val)
                marginal_probs_x[val], marginal_probs_y[val] = num_samples_x / num_samples, num_samples_y / num_samples

            joint_probs_xy = np.zeros((self.num_states, self.num_states))
            for val_x in range(self.num_states):
                for val_y in range(self.num_states):
                    num_cur_samples = sum(np.logical_and(samples_x == val_x, samples_y == val_y))
                    joint_probs_xy[val_x, val_y] = num_cur_samples / num_samples
            mi_matrix[x, y] = mi_matrix[y, x] = self.compute_mi(marginals_x=marginal_probs_x,
                                                                marginals_y=marginal_probs_y,
                                                                joints_xy=joint_probs_xy)

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
   
  
