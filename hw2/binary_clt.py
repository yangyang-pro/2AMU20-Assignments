from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools

ROOT = -1


class BinaryCLT:

    # Task 2a and 2b
    def __init__(self, data, root=None, alpha=0.01):
        """
        Initialize and learn a binary CLT.
        :param data: The training data.
        :param root: The root RV, if None choose one randomly.
        :param alpha: Laplace smoothing parameter.
        """
        self.n_features = data.shape[1]
        self.alpha = alpha
        if root is None:
            root = np.random.choice(self.n_features)
        else:
            assert 0 <= root < self.n_features
        self.root = root

        priors, joints = self.__estimate_priors_joints(data, self.alpha)
        mutual_info = self.__estimate_mutual_information(priors, joints)
        max_st = minimum_spanning_tree(-(mutual_info + 1.0))

        bfs, tree = breadth_first_order(max_st, directed=False, i_start=self.root, return_predecessors=True)
        tree[self.root] = ROOT
        self.bfs = bfs.tolist()

        # CLT structure
        self.tree = tree.tolist()
        # CLT log parameters
        self.log_params = self.__estimate_clt_log_params(self.tree, priors, joints)

    # Task 2a
    def get_tree(self):
        return self.tree

    # Task 2b
    def get_log_params(self):
        return self.log_params

    # Task 2c
    def log_prob(self, x, exhaustive=False):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param exhaustive: True for exhaustive inference, False for message passing inference.
        :param x: The queries.
        :return: The resulting log likelihood.
        """

        x = x.copy()
        tree = self.tree
        log_params = self.log_params
        rvs = np.arange(self.n_features).tolist()

        if not np.isnan(x).any():
            # no marginalisation required (no difference between exhaustive inference or message passing)
            x = x.astype(np.int8)
            log_prob = np.sum(log_params[rvs, x[:, tree], x[:, rvs]], axis=1)

        elif exhaustive:
            # marginalisation required, use exhaustive inference
            if True:
                # first technique, faster but requires more space
                log_prob = np.zeros(x.shape[0])
                states = np.array([state for state in itertools.product([0, 1], repeat=self.n_features)])
                states_log_probs = np.sum(log_params[rvs, states[:, tree], states[:, rvs]], axis=1)
                for i in range(x.shape[0]):
                    non_nan_idx = np.where(~np.isnan(x[i]))[0]
                    mar_idx = np.where((states[:, non_nan_idx] == x[i][non_nan_idx]).all(axis=1))[0]
                    log_prob[i] = logsumexp(states_log_probs[mar_idx])
            else:
                # second technique, slower (about 7 minutes on marginals NLTCS) but requires less space
                log_prob = np.zeros(x.shape[0])
                for i in range(x.shape[0]):
                    nan_idx = np.where(np.isnan(x[i]))[0]
                    states = np.array([state for state in itertools.product([0, 1], repeat=nan_idx.shape[0])])
                    states_log_prob = np.zeros(mar_states.shape[0])
                    for j in range(mar_states.shape[0]):
                        x[i][nan_idx] = states[j]
                        states_log_prob[j] = np.sum(log_params[rvs, x[i][tree].astype(np.int8), x[i].astype(np.int8)])
                    log_prob[i] = logsumexp(states_log_prob)
        else:
            # marginalisation required, use message passing
            log_prob = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                messages = np.zeros((self.n_features, 2))
                # Let's proceed bottom-up, use the reversed bfs [::-1] and exclude the root node [:-1]
                for j in self.bfs[::-1][:-1]:
                    # If non-observed value then factor marginalize that variable
                    if not np.isnan(x[i, j]):
                        messages[tree[j], 0] += log_params[j, 0, int(x[i, j])] + messages[j, int(x[i, j])]
                        messages[tree[j], 1] += log_params[j, 1, int(x[i, j])] + messages[j, int(x[i, j])]
                    else:
                        messages[tree[j], 0] += logsumexp([
                            log_params[j, 0, 0] + messages[j, 0],
                            log_params[j, 0, 1] + messages[j, 1]])
                        messages[tree[j], 1] += logsumexp([
                            log_params[j, 1, 0] + messages[j, 0],
                            log_params[j, 1, 1] + messages[j, 1]])

                # Compute the final likelihood considering the root node
                # Note that self.params[self.root, 1] = self.params[self.root], since it is unconditioned
                if not np.isnan(x[i, self.root]):
                    # the root RV is observed
                    log_prob[i] = log_params[self.root, 0, int(x[i, self.root])] + messages[self.root, int(x[i, self.root])]
                else:
                    # the root RV is not observed
                    log_prob[i] = logsumexp([
                        log_params[self.root, 0, 0] + messages[self.root, 0],
                        log_params[self.root, 0, 1] + messages[self.root, 1]])

        return np.expand_dims(log_prob, axis=1)

    # Task 2d
    def sample(self, n_samples):
        """
        Sample from the CLT.

        :param n_samples: The number of desirable samples.
        :return: The  samples.
        """

        samples = np.ones((n_samples, self.n_features)).astype(dtype=np.uint8)
        p_root = np.exp(self.log_params[self.root, 0, 1])
        # sample the root node first
        samples[:, self.root] = np.random.binomial(1, p_root, n_samples)

        for feature in self.bfs[1:]:
            parent_sample = samples[:, self.tree[feature]]
            p_parent = np.exp(self.log_params[feature][parent_sample])[:, 1]
            samples[:, feature] = np.random.binomial(1, p_parent, n_samples)

        return samples

    @staticmethod
    def __estimate_priors_joints(data, alpha):
        """
        Estimate both priors and joints probability distributions from data via maximum likelihood estimation.

        :param data: The binary data.
        :param alpha: Laplace smoothing factor.
        :return: A pair of priors and joints distributions.
                 Note that priors[i, k] := P(X_i=k).
                 Note that joints[i, j, k, l] := P(X_i=k, X_j=l).
        """

        # float32 allows fast dot product
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        n_features = data.shape[1]
        n_samples = data.shape[0]

        # ones[i, j] = 1[data[:, i]=1, data[:, j]=1]
        ones = np.dot(data.T, data).astype(np.float64)
        # ones_diag[i] = 1[data[:, i]=1]
        ones_diag = np.diag(ones)

        priors = np.zeros((n_features, 2))
        priors[:, 1] = (ones_diag + 2 * alpha) / (n_samples + 4 * alpha)
        priors[:, 0] = 1 - priors[:, 1]

        # cols_diag[i, j] = 1[data[:, j]=1], for every i
        diag_c = ones_diag * np.ones((n_features, n_features))
        # cols_diag[i, j] = 1[data[:, i]=1], for every j
        diag_r = diag_c.transpose()

        joints = np.zeros((n_features, n_features, 2, 2))
        # joints[i, j, 0, 0] := p(X_i=0, X_j=0)
        joints[:, :, 0, 0] = n_samples - diag_c - diag_r + ones
        # joints[i, j, 0, 1] := p(X_i=0, X_j=1)
        joints[:, :, 0, 1] = diag_c - ones
        # joints[i, j, 1, 0] := p(X_i=1, X_j=0)
        joints[:, :, 1, 0] = diag_r - ones
        # joints[i, j, 1, 1] := p(X_i=1, X_j=1)
        joints[:, :, 1, 1] = ones
        joints = (joints + alpha) / (n_samples + 4 * alpha)

        return priors, joints

    @staticmethod
    def __estimate_mutual_information(priors, joints):
        """
        Estimate the mutual information matrix, given priors and joints distributions via maximum likelihood estimation.

        :param priors: The priors probability distributions, i.e. priors[i, k] := P(X_i=k).
        :param joints: The joints probability distributions, i.e. joints[i, j, k, l] := P(X_i=k, X_j=l).
        :return: The mutual information matrix, i.e. mutual_info[i,j] = MI(X_i, X_j).
        """
        # marginal factorizations
        mar_facts = np.zeros((priors.shape[0], priors.shape[0], 2, 2))
        # outers[i, j, 0, 0] := P(X_i=0) * P(X_j=0)
        mar_facts[:, :, 0, 0] = np.outer(priors[:, 0], priors[:, 0])
        # outers[i, j, 0, 1] := P(X_i=0) * P(X_j=1)
        mar_facts[:, :, 0, 1] = np.outer(priors[:, 0], priors[:, 1])
        # outers[i, j, 1, 0] := P(X_i=1) * P(X_j=0)
        mar_facts[:, :, 1, 0] = np.outer(priors[:, 1], priors[:, 0])
        # outers[i, j, 1, 1] := P(X_i=1) * P(X_j=1)
        mar_facts[:, :, 1, 1] = np.outer(priors[:, 1], priors[:, 1])

        mutual_info = np.sum(joints * (np.log(joints) - np.log(mar_facts)), axis=(2, 3))
        np.fill_diagonal(mutual_info, 0)

        return mutual_info

    @staticmethod
    def __estimate_clt_log_params(tree, priors, joints):
        """
        Estimate the parameters of a CLT.

        :param tree: The tree structure, i.e. a list of predecessors in a tree structure.
        :param priors: The priors distributions, s.t. priors[i, k] := P(X_i=k).
        :param joints: The joints distributions, s.t. joints[i, j, k, l] := P(X_i=k, X_j=l).
        :return: The log conditional probability tables (CPTs) in a tensorized form.
                 Note that log_params[i, l, k] = log P(X_i=k | Pa(X_i)=l).
                 A special case is made for the root distribution which is not conditioned.
                 Note that log_params[root, :, k] = log P(X_root=k).
        """
        log_param = np.zeros((priors.shape[0], 2, 2))
        root = tree.index(ROOT)

        features = np.arange(priors.shape[0]).tolist()
        features.remove(root)

        parents = tree.copy()
        parents.pop(root)

        log_priors = np.log(priors)
        log_param[root, 0, 0] = log_param[root, 1, 0] = log_priors[root, 0]
        log_param[root, 0, 1] = log_param[root, 1, 1] = log_priors[root, 1]

        log_joints = np.log(joints)
        log_param[features, 0, 0] = log_joints[features, parents, 0, 0] - log_priors[parents, 0]
        log_param[features, 1, 0] = log_joints[features, parents, 0, 1] - log_priors[parents, 1]
        log_param[features, 0, 1] = log_joints[features, parents, 1, 0] - log_priors[parents, 0]
        log_param[features, 1, 1] = log_joints[features, parents, 1, 1] - log_priors[parents, 1]

        return log_param
