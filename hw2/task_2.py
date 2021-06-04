import numpy as np
import csv
import itertools
import networkx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
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
        # labels of random variables
        self.rvs = np.arange(self.num_rvs)
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
        """
        compute the mutual information of two random variables
        :param marginals_x: the marginal probability of rv x
        :param marginals_y: the marginal probability of rv y
        :param joints_xy: the joint probability of rv x and rv y
        :return: the mutual information value
        """
        mi = 0
        for val_x in range(self.num_states):
            for val_y in range(self.num_states):
                mi += joints_xy[val_x, val_y] * np.log(joints_xy[val_x, val_y] /
                                                       (marginals_x[val_x] * marginals_y[val_y]))
        return mi

    def create_clt(self):
        """
        construct the Chow-Liu Trees (CLT) based on the input data
        :return: an undirected CLT stored as a sparse matrix
        """
        # the mutual information matrix stores the mi value between every two random variables
        mi_matrix = np.zeros((self.num_rvs, self.num_rvs))
        # consider every combination of two random variables
        rv_combinations = list(itertools.combinations(self.rvs, 2))
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
                marginal_probs_x[val] = (2 * self.alpha + num_samples_x) / (4 * self.alpha + self.num_samples)
                marginal_probs_y[val] = (2 * self.alpha + num_samples_y) / (4 * self.alpha + self.num_samples)

            # calculate the joint probabilities of x and y by frequencies
            joint_probs_xy = np.zeros((self.num_states, self.num_states))
            for val_x in range(self.num_states):
                for val_y in range(self.num_states):
                    num_joint_samples = sum(np.logical_and(samples_x == val_x, samples_y == val_y))
                    joint_probs_xy[val_x, val_y] = (self.alpha + num_joint_samples) / \
                                                   (4 * self.alpha + self.num_samples)
            mi_matrix[x, y] = self.compute_mi(marginals_x=marginal_probs_x,
                                              marginals_y=marginal_probs_y,
                                              joints_xy=joint_probs_xy)
        # the mutual information matrix represents a fully-connected network,
        # we calculate the maximum spanning tree as our Chow-Liu tree
        # maximum_spanning_tree(mi) = minimum_spanning_tree(-mi),
        # since the mi between nodes is possibly very small, plus 1 here to make sure every nodes is connected
        maximum_spanning_tree = minimum_spanning_tree(-(mi_matrix + 1))
        return maximum_spanning_tree

    def get_tree(self):
        """
        compute the structure of the clt by applying breadth-first order starting from the root node
        :return: a list of predecessors of the learned structure:
        If X_j is the parent of X_i then tree[i] = j, while,
        if X_i is the root of the tree then tree[i] = -1.
        """
        self.bfo, predecessors = breadth_first_order(self.clt, i_start=self.root, directed=False)
        predecessors[self.root] = -1
        self.predecessors = predecessors
        return self.predecessors

    def plot_tree(self):
        """
        make a plot of the constructed CLT
        :return:
        """
        if self.predecessors is None:
            _ = self.get_tree()
        G = networkx.DiGraph()
        for i in range(1, self.predecessors.shape[0]):
            G.add_edge(self.predecessors[i], i)
        pos = graphviz_layout(G, prog='dot')
        networkx.draw(G, pos, with_labels=True)
        plt.show()

    def get_log_params(self):
        """
        compute the conditional probabilities table (CPT)
        :return: a (D, 2, 2)-dimensional array such that log params[i,j,k] = log p(x_i = k | x_t(i) = j),
        where D is the number of RVs and t(i) is the index of X_i's parent.
        """
        log_params = np.zeros((self.num_rvs, self.num_states, self.num_states))
        # treat the root node as a special case since it has no parent
        # calculate the probabilities for the root node
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
                # estimate the probabilities by maximum likelihood and using Laplace's correction.
                prob_parent = (2 * self.alpha + num_parent_samples) / (4 * self.alpha + self.num_samples)
                for val_cur in range(self.num_states):
                    num_joint_samples = sum(np.logical_and(samples_cur == val_cur, samples_parent == val_parent))
                    prob_joint = (self.alpha + num_joint_samples) / (4 * self.alpha + self.num_samples)
                    log_params[cur, val_parent, val_cur] = np.log(prob_joint / prob_parent)
        self.log_params = log_params
        return log_params

    def log_prob(self, x, exhaustive=True):
        """
        Task 2c Inference
        :param x: the marginal queries
        :param exhaustive: boolean. Perform exhaustive inference or efficient inference based on message passing
        :return: the (N, 1)-dimensional float array lp such that lp[i] contains the log probability of x[i].
        """
        num_queries = x.shape[0]
        lp = np.zeros(num_queries)
        # exhaustive inference
        if exhaustive:
            for i in tqdm(range(num_queries)):
                # current query
                query = x[i]
                num_missing_rvs = sum(np.isnan(query))
                # if number of missing random variables is 0, we just calculate the joint probability of the query
                if num_missing_rvs == 0:
                    joint_prob = self.log_params[self.root, 0, int(query[self.root])]
                    for rv in self.bfo[1:]:
                        parent = self.predecessors[rv]
                        joint_prob += self.log_params[rv, int(query[parent]), int(query[rv])]
                    lp[i] = joint_prob
                # otherwise we need to calculate the marginal probabilities of known random variables
                else:
                    # to get marginals, we need to consider every combinations of values of the random variables
                    missing_rv_val_combinations = list(itertools.product(range(self.num_states),
                                                                         repeat=num_missing_rvs))
                    marginal_observed = []
                    for missing_rv_vals in missing_rv_val_combinations:
                        # for each combination, we calculate the corresponding joint probability
                        # we construct a masked query to calculate the joint more easily
                        masked_query = np.copy(query)
                        masked_query[np.isnan(masked_query)] = missing_rv_vals
                        joint_prob = self.log_params[self.root, 0, int(masked_query[self.root])]
                        for rv in self.bfo[1:]:
                            parent = self.predecessors[rv]
                            joint_prob += self.log_params[rv, int(masked_query[parent]), int(masked_query[rv])]
                        marginal_observed.append(joint_prob)
                    lp[i] = logsumexp(marginal_observed)
        # efficient inference based on message passing
        else:
            # every non-root node will pass messages to its parent
            # Every node can send a message if and only if it has received messages from all its children
            # so we use a reversed topological order (reversed breadth-first order is exactly the same)
            reversed_bfo = self.bfo[::-1]
            for i in tqdm(range(num_queries)):
                # current query
                query = x[i]
                # initialize a message array (num_rvs, 2) with 0 to store the messages for each random variable
                messages = np.zeros((self.num_rvs, self.num_states))
                for rv in reversed_bfo[:-1]:
                    # recompute the message array to collect message from the children nodes of current node
                    messages = self.message_passing(messages=messages, rv=rv, query=query)
                # treat the root node as a special case since it has no parent
                root_children = self.rvs[self.predecessors == self.root]
                # check if the value of the root node is known in the query
                if np.isnan(query[self.root]):
                    marginal_root = 0
                    for val_root in range(self.num_states):
                        prob_root = np.exp(self.log_params[self.root, 0, val_root])
                        # collect message from the children nodes
                        for child in root_children:
                            prob_root *= messages[child, val_root]
                        marginal_root += prob_root
                    lp[i] = np.log(marginal_root)
                else:
                    val_root = int(query[self.root])
                    prob_root = np.exp(self.log_params[self.root, 0, val_root])
                    # collect message from the children nodes
                    for child in root_children:
                        prob_root *= messages[child, val_root]
                    lp[i] = np.log(prob_root)
        print(np.sum(np.exp(lp)))
        return lp

    def message_passing(self, messages, rv, query):
        """
        passing the message from the children nodes to current node
        :param messages: an array stores the messages for all the nodes
        :param rv: current random variable
        :param query: current marginal query
        :return: modified messages
        """
        # get the label of children of the current node
        rv_children = self.rvs[self.predecessors == rv]
        # get the label of parent of the current node
        rv_parent = self.predecessors[rv]
        # get the value of current node and its parent
        val_rv = query[rv]
        val_rv_parent = query[rv_parent]
        # check if values of current node and parent node are known in the query
        # follow the formula of computing the messages
        if np.isnan(val_rv) and np.isnan(val_rv_parent):
            for vp in range(self.num_states):
                message_vp = 0
                for vr in range(self.num_states):
                    # get the probability of the current node
                    prob_rv = np.exp(self.log_params[rv, vp, vr])
                    # pass the message from the children nodes to current node
                    for child in rv_children:
                        prob_rv *= messages[child, vr]
                    message_vp += prob_rv
                messages[rv, vp] = message_vp
        elif np.isnan(val_rv) and not np.isnan(val_rv_parent):
            val_rv_parent = int(val_rv_parent)
            message_vp = 0
            for vr in range(self.num_states):
                # get the probability of the current node
                prob_rv = np.exp(self.log_params[rv, val_rv_parent, vr])
                # pass the message from the children nodes to current node
                for child in rv_children:
                    prob_rv *= messages[child, vr]
                message_vp += prob_rv
            messages[rv, val_rv_parent] = message_vp
        elif not np.isnan(val_rv) and np.isnan(val_rv_parent):
            val_rv = int(val_rv)
            for vp in range(self.num_states):
                message_vp = 0
                # get the probability of the current node
                prob_rv = np.exp(self.log_params[rv, vp, val_rv])
                # pass the message from the children nodes to current node
                for child in rv_children:
                    prob_rv *= messages[child, val_rv]
                message_vp += prob_rv
                messages[rv, vp] = message_vp
        else:
            val_rv, val_rv_parent = int(val_rv), int(val_rv_parent)
            message_vp = 0
            # get the probability of the current node
            prob_rv = np.exp(self.log_params[rv, val_rv_parent, val_rv])
            # pass the message from the children nodes to current node
            for child in rv_children:
                prob_rv *= messages[child, val_rv]
            message_vp += prob_rv
            messages[rv, val_rv_parent] = message_vp
        return messages

    def sample(self, n_samples):
        """
        Ancestral sampling
        :param n_samples: number of generated samples
        :return: a sample matrix with shape (n_samples, num_rvs)
        """
        samples = np.zeros((n_samples, self.num_rvs))
        for i in tqdm(range(n_samples)):
            # we treat the root node as a special case since it has no parent
            root = self.root
            # get the probability of root node from the CPT
            prob_root = np.exp(self.log_params[root, 0, 1])
            # generate a random number between 0 and 1
            random_number = np.random.uniform(0, 1)
            # compare the random number with the probability of the root node
            samples[i, root] = 1 if random_number <= prob_root else 0
            # for the non-root node, we get its conditional probability
            for rv in self.bfo[1:]:
                # since we follow breadth-first oder (top-down),
                # we already know the state of the parent of current node
                rv_parent = self.predecessors[rv]
                val_parent = int(samples[i, int(rv_parent)])
                prob_rv = np.exp(self.log_params[rv, val_parent, 1])
                random_number = np.random.uniform(0, 1)
                samples[i, rv] = 1 if random_number <= prob_rv else 0
        return samples


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
    # print(clt.log_prob(queries, exhaustive=True))
    print(clt.log_prob(queries, exhaustive=False))