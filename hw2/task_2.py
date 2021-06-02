from numpy.random import triangular
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
import itertools
import networkx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from tqdm import tqdm


class BinaryCLT:
    def __init__(self, data, root=0, alpha=0.01):
        self.data = data
        self.root = root
        self.alpha = alpha
        # for binary discrete random variables, the number of states is 2: [0, 1]
        self.num_states = 2
        # number of samples from the real distribution
        self.num_samples = data.shape[0]
        # number of random variables
        self.num_rvs = data.shape[1]
        # Chow-Liu trees, stored as a sparse matrix
        self.clt = self.create_clt()
        # list of predecessors of the learned structure: If X_j is the parent of X_i then tree[i] =
        # j, while, if X_i is the root of the tree then tree[i] = -1.
        self.predecessors = None
        # breadth first order of the clt
        self.bfo = None
        # conditional probabilities table of the clt
        self.log_params = None

    def compute_mi(self, marginals_x, marginals_y, joints_xy):
        mi = 0
        for val_x in range(self.num_states):
            for val_y in range(self.num_states):
                mi += joints_xy[val_x, val_y] * np.log(joints_xy[val_x, val_y] /
                                                       (marginals_x[val_x] * marginals_y[val_y]))
        return mi

    def create_clt(self):
        rvs = np.arange(self.num_rvs)
        rv_combinations = list(itertools.combinations(rvs, 2))
        mi_matrix = np.zeros((self.num_rvs, self.num_rvs))
        for rv_combination in tqdm(rv_combinations):
            # current random variables
            x, y = rv_combination
            # samples for current random variables
            samples_xy = self.data[:, rv_combination]
            samples_x = samples_xy[:, 0]
            samples_y = samples_xy[:, 1]

            # calculate the marginal probabilities of x and y by frequencies
            marginal_probs_x = np.zeros(self.num_states)
            marginal_probs_y = np.zeros(self.num_states)
            for val in range(self.num_states):
                num_samples_x, num_samples_y = sum(samples_x == val), sum(samples_y == val)
                marginal_probs_x[val] = num_samples_x / self.num_samples
                marginal_probs_y[val] = num_samples_y / self.num_samples

            # calculate the joint probabilities of x and y by frequencies
            joint_probs_xy = np.zeros((self.num_states, self.num_states))
            for val_x in range(self.num_states):
                for val_y in range(self.num_states):
                    num_joint_samples = sum(np.logical_and(samples_x == val_x, samples_y == val_y))
                    joint_probs_xy[val_x, val_y] = num_joint_samples / self.num_samples
            mi_matrix[x, y] = self.compute_mi(marginals_x=marginal_probs_x,
                                              marginals_y=marginal_probs_y,
                                              joints_xy=joint_probs_xy)
        maximum_spanning_tree = minimum_spanning_tree(-(mi_matrix + 1))
        return maximum_spanning_tree

    def get_tree(self):
        self.bfo, predecessors = breadth_first_order(self.clt, i_start=self.root, directed=False)
        predecessors[self.root] = -1
        self.predecessors = predecessors
        return self.predecessors

    def plot_tree(self):
        if self.predecessors is None:
            _ = self.get_tree()
        G = networkx.DiGraph()
        for i in range(1, self.predecessors.shape[0]):
            G.add_edge(self.predecessors[i], i)
        pos = graphviz_layout(G, prog='dot')
        networkx.draw(G, pos, with_labels=True)
        plt.show()

    def get_log_params(self):
        log_params = np.zeros((self.num_rvs, self.num_states, self.num_states))
        # calculate the probabilities for root node (no parent)
        samples_root = self.data[:, self.root]
        for val_root in range(self.num_states):
            num_samples_root = sum(samples_root == val_root)
            prob_root = num_samples_root / self.num_samples
            for i in range(self.num_states):
                log_params[self.root, i, val_root] = np.log(prob_root)
        # calculate the conditional probabilities based on the breadth-first order of the clt (top-down)
        # skip the root node
        for cur in self.bfo[1:]:
            # samples for current node
            samples_cur = self.data[:, cur]
            # parent node of current node
            parent = self.predecessors[cur]
            samples_parent = self.data[:, parent]
            for val_parent in range(self.num_states):
                num_parent_samples = sum(samples_parent == val_parent)
                prob_parent = (2 * self.alpha + num_parent_samples) / (4 * self.alpha + self.num_samples)
                for val_cur in range(self.num_states):
                    num_joint_samples = sum(np.logical_and(samples_cur == val_cur, samples_parent == val_parent))
                    prob_joint = (self.alpha + num_joint_samples) / (4 * self.alpha + self.num_samples)
                    log_params[cur, val_parent, val_cur] = np.log(prob_joint / prob_parent)
        self.log_params = log_params
        return log_params

    def log_prob(self, x, exhaustive=True):
        num_queries = x.shape[0]
        log_probs = np.zeros(num_queries)
        # exhaustive inference
        if exhaustive:
            for i in tqdm(range(num_queries)):
                query = x[i]
                num_missing_rvs = sum(np.isnan(query))
                if num_missing_rvs == 0:
                    # log_probs_query = []
                    lp = 0
                    for rv in range(self.num_rvs):
                        parent = self.predecessors[rv]
                        if parent == -1:
                            # log_probs_query.append(self.log_params[rv, 0, int(query[rv])])
                            lp += self.log_params[rv, 0, int(query[rv])]
                        else:
                            # log_probs_query.append(self.log_params[rv, int(query[parent]), int(query[rv])])
                            lp += self.log_params[rv, int(query[parent]), int(query[rv])]
                    # log_probs[i] = logsumexp(log_probs_query)
                    # log_probs[i] = np.log(np.prod(np.exp(log_probs_query)))
                    log_probs[i] = lp
                else:
                    missing_rv_val_combinations = list(itertools.product(range(self.num_states), repeat=num_missing_rvs))
                    marginal_observed = []
                    for missing_rv_vals in missing_rv_val_combinations:
                        masked_query = np.copy(query)
                        masked_query[np.isnan(masked_query)] = missing_rv_vals
                        # log_probs_query = []
                        lp = 0
                        for rv in range(self.num_rvs):
                            parent = self.predecessors[rv]
                            if parent == -1:
                                # log_probs_query.append(self.log_params[rv, 0, int(masked_query[rv])])
                                lp += self.log_params[rv, 0, int(masked_query[rv])]
                            else:
                                # log_probs_query.append(self.log_params[rv, int(masked_query[parent]), int(masked_query[rv])])
                                lp += self.log_params[rv, int(masked_query[parent]), int(masked_query[rv])]
                        # marginal_observed += logsumexp(log_probs_query)
                        # marginal_observed += np.log(np.prod(np.exp(log_probs_query)))
                        print(np.exp(lp))
                        marginal_observed.append(lp)
                    log_probs[i] = np.log(np.sum(np.exp(marginal_observed)))
        else:
            pass
        print(np.sum(np.exp(log_probs)))
        return log_probs

    def sample(self, n_samples):
        pass


if __name__ == "__main__":
    with open('nltcs.train.data', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        train = np.array(list(reader)).astype(np.float)

    with open('nltcs.test.data', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        test = np.array(list(reader)).astype(np.float)

    with open('nltcs.valid.data', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        valid = np.array(list(reader)).astype(np.float)

    with open('nltcs_marginals.data', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        queries = np.array(list(reader)).astype(np.float)

    clt = BinaryCLT(train)
    print(clt.get_tree())
    clt.plot_tree()
    print(clt.get_log_params())
    print(clt.log_prob(queries, exhaustive=True))