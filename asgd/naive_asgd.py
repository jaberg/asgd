"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""

import sys
import numpy as np
from numpy import dot
from itertools import izip

DEFAULT_SGD_STEP_SIZE0 = 1e-2
DEFAULT_L2_REGULARIZATION = 1e-3
DEFAULT_MIN_OBSERVATIONS = 1000
DEFAULT_MAX_OBSERVATIONS = sys.maxint
DEFAULT_FEEDBACK = False
DEFAULT_RSTATE = None
DEFAULT_DTYPE = np.float32
DEFAULT_N_PARTIAL = 5000
DEFAULT_FIT_ABS_TOLERANCE = 1e-4
DEFAULT_FIT_REL_TOLERANCE = 1e-2


class BaseASGD(object):

    def __init__(self, n_features,
                 sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 min_observations=DEFAULT_MIN_OBSERVATIONS,
                 max_observations=DEFAULT_MAX_OBSERVATIONS,
                 fit_n_partial=DEFAULT_N_PARTIAL,
                 fit_abs_tolerance=DEFAULT_FIT_ABS_TOLERANCE,
                 fit_rel_tolerance=DEFAULT_FIT_REL_TOLERANCE,
                 feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE,
                 dtype=DEFAULT_DTYPE):

        # --
        assert n_features > 1
        self.n_features = n_features

        if not 0 <= min_observations <= max_observations:
            raise ValueError('min_observations > max_observations',
                    (min_observations, max_observations))
        self.min_observations = min_observations
        self.max_observations = max_observations

        self.fit_n_partial = fit_n_partial
        self.fit_abs_tolerance = fit_abs_tolerance
        self.fit_rel_tolerance = fit_rel_tolerance

        if feedback:
            raise NotImplementedError("FIXME: feedback support is buggy")
        self.feedback = feedback

        if rstate is None:
            rstate = np.random.RandomState()
        self.rstate = rstate

        assert l2_regularization > 0
        self.l2_regularization = l2_regularization
        self.dtype = dtype

        # --
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = 2. / 3
        self.sgd_step_size_scheduling_multiplier = l2_regularization

        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0

        # --
        self.n_observations = 0
        self.train_means = []
        self.recent_train_costs = []

    def fit_converged(self):
        """
        There are two convergence tests here. Training is considered to be
        over if it has

        * stalled: latest train_means is allclose to a previous one (*)

        * terminated: latest train_means is allclose to 0

        """
        train_means = self.train_means
        rtol = self.fit_rel_tolerance
        atol = self.fit_abs_tolerance

        # -- check for perfect fit
        if len(train_means) > 1:
            assert np.min(train_means) >= 0
            if np.allclose(train_means[-1], 0, atol=atol, rtol=rtol):
                return True

        # -- check for stall condition
        if len(train_means) > 2:
            old_pt = max(
                    len(train_means) // 2,
                    len(train_means) - 4)
            thresh = (1 - rtol) * train_means[old_pt] - atol
            if train_means[-1] > thresh:
                return True

        return False

    def fit(self, X, y):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_points_remaining = self.max_observations - self.n_observations

        while n_points_remaining > 0:

            # -- every iteration will train from n_partial observations and
            # then check for convergence
            fit_n_partial = min(n_points_remaining, self.fit_n_partial)

            idx = self.rstate.permutation(n_points)
            Xb = X[idx[:fit_n_partial]]
            yb = y[idx[:fit_n_partial]]
            self.partial_fit(Xb, yb)

            if self.feedback:
                raise NotImplementedError(
                    'partial_fit logic requires memory to be distinct')
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

            if (self.n_observations >= self.min_observations
                    and self.fit_converged()):
                break

            n_points_remaining -= len(Xb)

        return self


class BaseBinaryASGD(BaseASGD):

    def __init__(self, *args, **kwargs):
        BaseASGD.__init__(self, *args, **kwargs)

        self.sgd_weights = np.zeros((self.n_features,), dtype=self.dtype)
        self.sgd_bias = np.asarray(0, dtype=self.dtype)

        self.asgd_weights = np.zeros((self.n_features,), dtype=self.dtype)
        self.asgd_bias = np.asarray(0, dtype=self.dtype)

    def decision_function(self, X):
        X = np.asarray(X)
        return dot(self.asgd_weights, X.T) + self.asgd_bias

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def cost(self, X, y):
        weights = self.asgd_weights
        bias = self.asgd_bias
        margin = y * (np.dot(X, weights) + bias)
        l2_cost = self.l2_regularization * (weights ** 2).sum()
        rval = np.maximum(0, 1 - margin).mean() + l2_cost
        return rval


class NaiveBinaryASGD(BaseBinaryASGD):

    def partial_fit(self, X, y):

        assert np.all(y**2 == 1)

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        train_means = self.train_means
        recent_train_costs = self.recent_train_costs

        for obs, label in izip(X, y):

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            if margin < 1:

                sgd_weights += sgd_step_size * label * obs
                sgd_bias += sgd_step_size * label
                recent_train_costs.append(1 - float(margin))
            else:
                recent_train_costs.append(0)

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # 4.1 update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

            if len(recent_train_costs) == self.fit_n_partial:
                train_means.append(np.mean(recent_train_costs)
                        + l2_regularization * np.dot(
                            self.asgd_weights, self.asgd_weights))
                self.recent_train_costs = recent_train_costs = []

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        return self


class BaseMultiASGD(BaseASGD):
    """
    Allocate weight vectors for one-vs-all multiclass SVM
    """
    def __init__(self, n_classes, n_features, *args, **kwargs):
        BaseASGD.__init__(self, n_features, *args, **kwargs)

        # --
        self.n_classes = n_classes

        dtype = self.dtype
        # --
        self.sgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.sgd_bias = np.zeros(n_classes, dtype=dtype)
        self.asgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.asgd_bias = np.zeros(n_classes, dtype=dtype)

    def decision_function(self, X):
        X = np.asarray(X)
        return dot(X, self.asgd_weights) + self.asgd_bias

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)


class NaiveOVAASGD(BaseMultiASGD):

    def partial_fit(self, X, y):

        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        n_classes = self.n_classes

        train_means = self.train_means
        recent_train_costs = self.recent_train_costs

        for obs, label in izip(X, y):
            label = 2 * (np.arange(n_classes) == label).astype(int) - 1

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            violations = margin < 1

            if violations:
                label_violated = label[violations]
                sgd_weights[:, violations] += (
                    sgd_step_size
                    * label_violated[np.newaxis, :]
                    * obs[:, np.newaxis]
                )
                sgd_bias[violations] += sgd_step_size * label_violated
                recent_train_costs.append(
                        n_classes - margin[violations].sum())
            else:
                recent_train_costs.append(0.0)

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # -- update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

            if len(recent_train_costs) == self.fit_n_partial:
                train_means.append(np.mean(recent_train_costs)
                        + l2_regularization * np.dot(
                            self.asgd_weights, self.asgd_weights))
                self.recent_train_costs = recent_train_costs = []

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        return self


class NaiveRankASGD(BaseMultiASGD):
    """
    Implements rank-based multiclass SVM.
    """

    def partial_fit(self, X, y):

        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        n_classes = self.n_classes

        train_means = self.train_means
        recent_train_costs = self.recent_train_costs

        for obs, label in izip(X, y):

            # -- compute margin
            decision = dot(obs, sgd_weights) + sgd_bias

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            dsrt = np.argsort(decision)
            distractor = dsrt[-2] if dsrt[-1] is label else dsrt[-1]
            margin = decision[label] - decision[distractor]

            if margin < 1:
                winc = sgd_step_size * obs
                sgd_weights[:, distractor] -= winc
                sgd_weights[:, label] += winc
                sgd_bias[distractor] -= sgd_step_size
                sgd_bias[label] += sgd_step_size
                recent_train_costs.append(1 - float(margin))
            else:
                recent_train_costs.append(0.0)

            # -- update asgd
            # -- TODO: delay updates until sgd_weights are modified,
            #          only 2 rows change per iteration
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # -- update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

            if len(recent_train_costs) == self.fit_n_partial:
                flat_weights = asgd_weights.flatten()
                l2_term = np.dot(flat_weights, flat_weights)
                train_means.append(np.mean(recent_train_costs)
                        + l2_regularization * l2_term)
                self.recent_train_costs = recent_train_costs = []

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        return self


    def cost(self, X, y):
        weights = self.asgd_weights
        bias = self.asgd_bias
        decisions = dot(X, weights) + bias
        margins = []
        for decision, label in zip(decisions, y):
            dsrt = np.argsort(decision)
            distractor = dsrt[-2] if dsrt[-1] is label else dsrt[-1]
            margin = decision[label] - decision[distractor]
            margins.append(margin)

        flat_weights = weights.flatten()
        l2_term = np.dot(flat_weights, flat_weights)
        l2_cost = self.l2_regularization * l2_term
        rval = np.maximum(0, 1 - np.asarray(margins)).mean() + l2_cost
        return rval
