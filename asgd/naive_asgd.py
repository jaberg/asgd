"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""

import copy
from itertools import izip
import sys
import time

import scipy.linalg
import numpy as np
from numpy import dot


from .base import BinaryClassifier
from .base import MultiClassifier

from .lossfns import loss_obj
from .lossfns import Hinge


DEFAULT_SGD_STEP_SIZE0 = None
DEFAULT_L2_REGULARIZATION = 1e-3
DEFAULT_MIN_OBSERVATIONS = 1000
DEFAULT_MAX_OBSERVATIONS = sys.maxint
DEFAULT_FEEDBACK = False

DEFAULT_RSTATE = 42
DEFAULT_DTYPE = np.float32
DEFAULT_N_PARTIAL = 5000
DEFAULT_FIT_ABS_TOLERANCE = 1e-4
DEFAULT_FIT_REL_TOLERANCE = 1e-2
DEFAULT_SGD_EXPONENT = 2.0 / 3.0
DEFAULT_SGD_TIMESCALE = 'l2_regularization'
DEFAULT_FIT_VERBOSE = False
# can be 'l2_regularization' or float
# This timescale default comes from [1] in which it is introduced as a
# heuristic.
# [1] http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
# Update: it is also recommended in Leon Bottou's SvmAsgd software.

DEFAULT_COST_FN = 'Hinge'

import line_profiler
if 0:
    profile = line_profiler.LineProfiler()
else:
    class DummyProfile(object):
        def print_stats(self):
            pass

        def __call__(self, f):
            return f
    profile = DummyProfile()


def assert_allclose(a, b, rtol=1e-05, atol=1e-08):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        adiff = abs(a - b).max(),
        rdiff = (abs(a - b) / (abs(a) + abs(b) + 1e-15)).max()
        raise ValueError('not close enough', (adiff, rdiff, {
            'amax': a.max(),
            'bmax': b.max(),
            'amin': a.min(),
            'bmin': b.min(),
            'asum': a.sum(),
            'bsum': b.sum(),
            }))


def fit_converged(train_means,
        atol=DEFAULT_FIT_ABS_TOLERANCE,
        rtol=DEFAULT_FIT_REL_TOLERANCE,
        verbose=DEFAULT_FIT_VERBOSE):
    """
    There are two convergence tests here. Training is considered to be
    over if it has

    * stalled: latest train_means is allclose to a previous one (*)

    * terminated: latest train_means is allclose to 0

    """

    if verbose:
        if train_means:
            print 'fit_converged:', len(train_means), train_means[-1]

    # -- check for perfect fit
    if len(train_means) > 1:
        assert np.min(train_means) >= 0
        if np.allclose(train_means[-1], 0, atol=atol, rtol=rtol):
            return True

    # -- check for stall condition
    if len(train_means) > 10:
        old_pt = max(
                len(train_means) // 2,
                len(train_means) - 10)
        thresh = (1 - rtol) * train_means[old_pt] - atol
        if train_means[-1] > thresh:
            return True

    return False


class BaseASGD(object):
    """
    XXX
    """

    def __init__(self, svm, data,
            sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            sgd_step_size_scheduling_exponent = DEFAULT_SGD_EXPONENT,
            sgd_step_size_scheduling_multiplier = DEFAULT_SGD_TIMESCALE,
            min_observations=DEFAULT_MIN_OBSERVATIONS,
            max_observations=DEFAULT_MAX_OBSERVATIONS,
            fit_n_partial=DEFAULT_N_PARTIAL,
            fit_abs_tolerance=DEFAULT_FIT_ABS_TOLERANCE,
            fit_rel_tolerance=DEFAULT_FIT_REL_TOLERANCE,
            fit_verbose=DEFAULT_FIT_VERBOSE,
            feedback=DEFAULT_FEEDBACK,
            rstate=DEFAULT_RSTATE,
            dtype=DEFAULT_DTYPE,
            cost_fn=DEFAULT_COST_FN,
            ):

        self.data = data
        self.svm = svm
        n_features = svm.n_features
        n_classes = svm.n_classes

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
        self.fit_verbose = fit_verbose

        self.cost_fn = loss_obj(cost_fn)

        if feedback:
            raise NotImplementedError("FIXME: feedback support is buggy")
        self.feedback = feedback

        if rstate is None:
            rstate = np.random.RandomState()
        elif type(rstate) is int:
            rstate = np.random.RandomState(rstate)
        self.rstate = rstate

        self.l2_regularization = l2_regularization
        self.dtype = dtype

        # --
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = \
            sgd_step_size_scheduling_exponent
        if sgd_step_size_scheduling_multiplier == 'l2_regularization':
            self.sgd_step_size_scheduling_multiplier = l2_regularization
        else:
            self.sgd_step_size_scheduling_multiplier = \
                    sgd_step_size_scheduling_multiplier

        # --
        self.n_observations = 0
        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0
        self.asgd_start = 0

        # --
        self.sgd_step_size = self.sgd_step_size0
        self.train_means = []
        self.recent_train_costs = []

        # --
        self.asgd_weights = svm.weights
        self.asgd_bias = svm.bias
        self.sgd_weights = svm.weights.copy()
        self.sgd_bias = svm.bias.copy()

    def __iter__(self):
        return self

    def next(self):
        X, y = self.data

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_points_remaining = self.max_observations - self.n_observations

        if n_points_remaining <= 0:
            raise StopIteration()

        # -- every iteration will train from n_partial observations and
        # then check for convergence
        fit_n_partial = min(n_points_remaining, self.fit_n_partial)

        all_idxs = self.rstate.permutation(n_points)
        idxs = all_idxs[:fit_n_partial]
        if hasattr(self, 'partial_fit_by_index'):
            self.partial_fit_by_index(idxs, X, y)
        else:
            self.partial_fit(X[idxs], y[idxs])

        if self.feedback:
            raise NotImplementedError(
                'partial_fit logic requires memory to be distinct')
            self.sgd_weights = self.asgd_weights
            self.sgd_bias = self.asgd_bias

        if (self.n_observations >= self.min_observations
                and fit_converged(self.train_means,
                    rtol=self.fit_rel_tolerance,
                    atol=self.fit_abs_tolerance,
                    verbose=self.fit_verbose)):
            raise StopIteration()

        self.svm.weights = self.asgd_weights
        self.svm.bias = self.asgd_bias

        return self.svm


class BinaryASGD(BaseASGD):
    """
    ASGD trainer for binary classification with any of the losses
    """

    def partial_fit(self, X, y):
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        if self.cost_fn.f != Hinge.f:
            raise NotImplementedError('Naive Binary requires hinge loss')

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

    def cost(self):
        X, y = self.data
        svm = self.svm
        margin = svm.decisions(X) * y
        assert margin.ndim == 1
        l2_cost = .5 * self.l2_regularization * (svm.weights ** 2).sum()
        loss = self.cost_fn.f(margin)
        rval = loss.mean() + l2_cost
        return rval


class OneVsAllASGD(BaseASGD):
    def partial_fit(self, X, y):
        return self.partial_fit_by_index(range(len(X)), X, y)

    @profile
    def partial_fit_by_index(self, idxs, X, y):
        ttt = time.time()
        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = np.asarray(self.sgd_weights, order='F')
        sgd_bias = self.sgd_bias

        asgd_weights = np.asarray(self.asgd_weights, order='F')
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size
        asgd_start = self.asgd_start

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        n_classes = self.n_classes
        n_features = asgd_weights.shape[0]

        train_means = self.train_means
        dtype = self.dtype
        recent_train_costs = self.recent_train_costs

        scal, axpy, gemv, gemm = scipy.linalg.blas.get_blas_funcs(
                ['scal', 'axpy', 'gemv', 'gemm'],
                (asgd_weights,))

        for idx in idxs:
            obs = X[idx].astype(dtype)
            yvec = -np.ones(n_classes, dtype=dtype)
            yvec[y[idx]] = 1

            # -- compute margin
            margin = yvec * gemv(alpha=1, a=sgd_weights, trans=1, x=obs,
                    beta=1.0, y=sgd_bias, overwrite_y=0,)
            #margin = yvec * (dot(obs, sgd_weights) + sgd_bias)
            #assert_allclose(margin, margin_)
            if asgd_start:
                asgd_margin = yvec * gemv(alpha=1, a=asgd_weights, trans=1, x=obs,
                        beta=1.0, y=asgd_bias, overwrite_y=0,)
                asgd_costs = self.cost_fn.f(asgd_margin)
                recent_train_costs.append(np.sum(asgd_costs))
            else:
                costs = self.cost_fn.f(margin)
                recent_train_costs.append(np.sum(costs))

            #asgd_margin = yvec * (dot(obs, asgd_weights) + asgd_bias)
            #assert_allclose(asgd_margin, asgd_margin_)

            # -- update sgd
            grad = self.cost_fn.df(margin) * yvec

            sgd_weights = gemm(
                    alpha=-sgd_step_size,
                    a=obs[:,None],
                    b=grad[None,:],
                    beta=1 - l2_regularization * sgd_step_size,
                    c=sgd_weights,
                    overwrite_c=1)
            sgd_bias = axpy(
                    a=-sgd_step_size,
                    x=grad,
                    y=sgd_bias)

            # -- update asgd
            if asgd_start:
                asgd_weights = axpy(a=asgd_step_size, x=sgd_weights,
                        y=scal(1 - asgd_step_size, asgd_weights))
                asgd_bias = axpy(a=asgd_step_size, x=sgd_bias,
                        y=scal(1 - asgd_step_size, asgd_bias))

            # -- update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)

            if asgd_start:
                asgd_step_size = 1. / max(1, n_observations - asgd_start)
            else:
                asgd_step_size = 1.

            if len(recent_train_costs) == self.fit_n_partial:
                if asgd_start:
                    weights = asgd_weights
                else:
                    weights = sgd_weights
                l2_term = np.dot(weights.flatten(), weights.flatten())
                train_means.append(np.mean(recent_train_costs)
                        + .5 * l2_regularization * l2_term)
                asgd_min_observations = 100000 # XXX
                if (asgd_start == 0 and
                        n_observations > (
                            self.min_observations - asgd_min_observations)):
                    asgd_start = n_observations
                if self.fit_verbose:
                    print 'train_mean', train_means[-1]
                    print 'l2_term', l2_term
                    print 'asgd_step_size', asgd_step_size
                # -- reset counter
                self.recent_train_costs = recent_train_costs = []

        if not asgd_start:
            asgd_weights = sgd_weights.copy()
            asgd_bias = sgd_bias.copy()

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size
        self.asgd_start = asgd_start

        self.n_observations = n_observations

        if self.fit_verbose:
            n_idx_per_sec = len(idxs) / (time.time() - ttt)
            print '%i observations at %f/sec' % (
                    n_observations, n_idx_per_sec)
        profile.print_stats()

        return self

    def cost(self):
        X, y = self.data
        svm = self.svm
        margin = svm.decisions(X) * (-1)
        margin[np.arange(len(y)), y] *= -1
        assert margin.ndim == 2
        l2_cost = .5 * self.l2_regularization * (svm.weights ** 2).sum()
        loss = self.cost_fn.f(margin)
        rval = loss.sum(axis=1).mean() + l2_cost
        return rval


class CrammerSingerCost(object):
    def cost(self):
        X, y = self.data
        svm = self.svm
        decisions = svm.decisions(X)
        margin = []
        for ii, (decision, label) in enumerate(zip(decisions, y)):
            dsrt = np.argsort(decision)
            distractor = dsrt[-2] if dsrt[-1] == label else dsrt[-1]
            margin.append(decision[label] - decision[distractor])
        l2_cost = .5 * self.l2_regularization * (svm.weights ** 2).sum()
        loss = self.cost_fn.f(np.asarray(margin))
        rval = loss.mean() + l2_cost
        return rval


class NaiveRankASGD(BaseASGD, CrammerSingerCost):
    """
    Implements rank-based multiclass SVM.
    """

    def partial_fit(self, X, y):
        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

        if self.cost_fn.f != Hinge.f:
            raise NotImplementedError('%s requires hinge loss' %
                    self.__class__.__name__)

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

            dsrt = np.argsort(decision)
            distractor = dsrt[-2] if dsrt[-1] == label else dsrt[-1]
            margin = decision[label] - decision[distractor]

            # -- update sgd
            if l2_regularization:
                sgd_weights *= 1 - l2_regularization * sgd_step_size

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

            asgd_start = self.min_observations // 2
            if n_observations <= asgd_start:
                asgd_step_size = 1.
            else:
                asgd_step_size = 1. / (n_observations - asgd_start)

            if len(recent_train_costs) == self.fit_n_partial:
                flat_weights = asgd_weights.flatten()
                l2_term = np.dot(flat_weights, flat_weights)
                l2_term += np.dot(asgd_bias, asgd_bias)
                train_means.append(np.mean(recent_train_costs)
                        + .5 * l2_regularization * l2_term)
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


class SparseUpdateRankASGD(BaseASGD, CrammerSingerCost):
    """
    Implements rank-based multiclass SVM.
    """

    #@profile
    def partial_fit(self, X, y):
        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")
        ttt = time.time()

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        # -- Fortran order is faster, and required for axpy
        sgd_weights = np.asarray(self.sgd_weights, order='F')
        sgd_bias = self.sgd_bias

        # -- Fortran order is faster, and required for axpy
        asgd_weights = np.asarray(self.asgd_weights, order='F')
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations
        n_classes = self.n_classes
        n_features = asgd_weights.shape[0]

        train_means = self.train_means
        recent_train_costs = self.recent_train_costs

        # -- the logical sgd_weight matrix is sgd_weights_scale * sgd_weights
        sgd_weights_scale = 1.0

        # -- the logical asgd_weights is stored as a row-wise linear
        # -- combination of asgd_weights and sgd_weights
        asgd_scale = np.ones((n_classes, 2), dtype=asgd_weights.dtype)
        asgd_scale[:, 1] = 0  # -- originally there is no sgd contribution

        scal, axpy, gemv = scipy.linalg.blas.get_blas_funcs(
                ['scal', 'axpy', 'gemv'],
                (asgd_weights,))

        for obs, label in izip(X, y):
            obs = obs.astype(sgd_weights.dtype)

            #print sgd_weights.shape, sgd_weights.strides
            #print asgd_weights.shape, asgd_weights.strides

            # -- compute margin
            decision = gemv(
                    alpha=sgd_weights_scale,
                    a=sgd_weights,
                    trans=1,
                    x=obs,
                    y=sgd_bias,
                    beta=1.0,
                    overwrite_y=0,
                    )

            dsrt = np.argsort(decision)
            distractor = dsrt[-2] if dsrt[-1] == label else dsrt[-1]
            margin = decision[label] - decision[distractor]

            # -- update sgd
            sgd_weights_scale *= 1 - l2_regularization * sgd_step_size

            cost = self.cost_fn.f(margin)
            grad = self.cost_fn.df(margin)

            if grad != 0.0:

                # -- perform the delayed updates to the rows of asgd
                for idx in [distractor, label]:
                    scal(
                            a=asgd_scale[idx, 0],
                            x=asgd_weights,
                            n=n_features,
                            offx=idx * n_features)
                    axpy(
                            a=(asgd_scale[idx, 1] *
                                sgd_weights_scale),
                            x=sgd_weights,
                            y=asgd_weights,
                            offx=idx * n_features,
                            offy=idx * n_features,
                            n=n_features)
                    asgd_scale[idx, 0] = 1
                    asgd_scale[idx, 1] = 0

                step = grad * sgd_step_size / sgd_weights_scale
                axpy(
                        a=step,
                        x=obs,
                        y=sgd_weights,
                        offy=distractor * n_features,
                        n=n_features)
                axpy(
                        a=-step,
                        x=obs,
                        y=sgd_weights,
                        offy=label * n_features,
                        n=n_features)
                sgd_bias[distractor] += grad * sgd_step_size
                sgd_bias[label] -= grad * sgd_step_size
                recent_train_costs.append(cost)
            else:
                recent_train_costs.append(cost)

            # -- update asgd via scale variables
            asgd_scale *= (1 - asgd_step_size)
            asgd_scale[:, 1] += asgd_step_size

            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # -- update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_start = self.min_observations // 2
            if n_observations <= asgd_start:
                asgd_step_size = 1.
            else:
                asgd_step_size = 1. / (n_observations - asgd_start)

            if len(recent_train_costs) == self.fit_n_partial:
                asgd_weights[:] *= asgd_scale[:, 0]
                asgd_weights[:] += (sgd_weights_scale * asgd_scale[:, 1]) * sgd_weights
                asgd_scale[:, 0] = 1
                asgd_scale[:, 1] = 0

                # -- Technically the stopping criterion should be based on
                #    the loss incurred by the asgd weights
                #    XXX: multiply entire minibatch by current asgd weights
                #    it won't take long
                flat_weights = asgd_weights.flatten()
                l2_term = np.dot(flat_weights, flat_weights)
                l2_term += np.dot(asgd_bias, asgd_bias)
                train_means.append(np.mean(recent_train_costs)
                        + l2_regularization * l2_term)
                self.recent_train_costs = recent_train_costs = []

        # --
        self.sgd_weights = sgd_weights * sgd_weights_scale
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        asgd_weights[:] *= asgd_scale[:, 0]
        asgd_weights[:] += (sgd_weights_scale * asgd_scale[:, 1]) * sgd_weights
        asgd_scale[:, 0] = 1
        asgd_scale[:, 1] = 0

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

        #profile.print_stats()
        if self.fit_verbose:
            print 'n_obs', n_observations, 'time/%i' % len(X), (time.time() - ttt)
        return self

