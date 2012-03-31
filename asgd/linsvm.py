"""
Automatic heuristic solver selection: LinearSVM

"""
import copy
import logging; logger = logging.getLogger(__name__)

import numpy as np

from .auto_step_size import binary_fit
from .naive_asgd import BinaryASGD
from .naive_asgd import OneVsAllASGD
from .naive_asgd import NaiveRankASGD
from .naive_asgd import SparseUpdateRankASGD

from .base import classifier

try:
    import sklearn.svm
except ImportError:
    pass

try:
    from .theano_asgd import TheanoOVA
except ImportError:
    pass


def find_sgd_step_size0(trainer_cls, svm, data, **kwargs):
    """XXX
    """

    def eval_size0(log2_size0):
        svm_copy = copy.deepcopy(svm)
        trainer = trainer_cls(svm_copy, data, **dict(kwargs,
            sgd_step_size0 = 2 ** log2_size0))
        trainer.next()  # train for one batch of examples
        rval = trainer.cost()
        if np.isnan(rval):
            rval = float('inf')
        logger.info('find step %e: %e' % (trainer.sgd_step_size0, rval))
        return rval

    tolerance = 1.0
    sgd_step_size0 = kwargs.get('sgd_step_size0', 1e-4)

    # N.B. we step downward first so that if both y0 == y1 == inf
    #      we stay going downward
    step = -tolerance
    x0 = np.log2(sgd_step_size0)
    x1 = np.log2(sgd_step_size0) + step
    y0 = eval_size0(x0)
    y1 = eval_size0(x1)
    if y1 > y0:
        step *= -1
        y0, y1 = y1, y0
        x0, x1 = x1, x0

    while (y1 < y0) or (y1 == float('inf')):
        x0, y0 = x1, y1
        x1 += step
        y1 = eval_size0(x1)

    # I tried using sp.optimize.fmin, but this function is bumpy and we only
    # want a really coarse estimate of the optimmum, so fmin and fmin_powell
    # end up being relatively inefficient even compared to this really stupid
    # search.
    #
    # TODO: increase the stepsize every time it still goes down, and then
    #       backtrack when we over-step

    rval = 2.0 ** x0
    return rval


class LinearSVM(object):
    """
    SVM-fitting object that implements a heuristic for choosing
    the right solver among several that may be installed in sklearn, and asgd.

    """

    def __init__(self, l2_regularization, solver='auto', label_dct=None,
            label_weights=None, verbose=False):
        self.l2_regularization = l2_regularization
        self.solver = solver
        self.label_dct = label_dct
        self.label_weights = label_weights
        self.verbose = verbose

    def fit(self, X, y):
        solver = self.solver
        label_dct = self.label_dct
        l2_regularization = self.l2_regularization

        if self.label_weights:
            raise NotImplementedError()

        n_train, n_features = X.shape
        if self.label_dct is None:
            label_dct = dict([(v, i) for (i, v) in enumerate(sorted(set(y)))])
        else:
            label_dct = self.label_dct
        n_classes = len(label_dct)

        if n_classes < 2:
            raise ValueError('must be at least 2 labels')

        elif n_classes == 2:
            if set(y) != set([-1, 1]):
                # TODO: use the label_dct to automatically adjust
                raise NotImplementedError()

            if solver == 'auto':
                if n_features > n_train:
                    solver = ('sklearn.svm.SVC', {'kernel': 'precomputed'})
                else:
                    solver = ('asgd.NaiveBinaryASGD', {})

            method, method_kwargs = solver

            if method in ('asgd.NaiveBinaryASGD', 'asgd.BinaryASGD'):
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                method_kwargs.setdefault('fit_verbose', self.verbose)
                auto_step_size0 = method_kwargs.pop(
                        'auto_step_size0', True)
                auto_max_examples = method_kwargs.pop(
                        'auto_max_examples', 1000)
                auto_step_size0_floor = method_kwargs.pop(
                        'auto_step_size0_floor', 1e-7)

                svm = classifier(n_classes=2, n_features=n_features,
                        dtype=method_kwargs.get('dtype', 'float32'))
                assert svm.weights.ndim == 1

                if auto_step_size0:
                    all_idxs = method_kwargs['rstate'].permutation(len(X))
                    idxs = all_idxs[:auto_max_examples]
                    step_size0 = find_sgd_step_size0(BinaryASGD,
                            svm=svm,
                            data=(X[idxs], y[idxs]),
                            l2_regularization=l2_regularization,
                            **method_kwargs)
                    step_size0 = max(step_size0 / 2.0, auto_step_size0_floor)
                    logger.info('setting sgd_step_size: %e' % step_size0)
                    method_kwargs['sgd_step_size0'] = float(step_size0)

                trainer = BinaryASGD(svm, (X, y),
                        l2_regularization=l2_regularization,
                        **method_kwargs)

                for svm in trainer:
                    # this loop does the training
                    pass

            elif method == 'sklearn.svm.SVC':
                C = 1.0 / (l2_regularization * len(X))
                svm = sklearn.svm.SVC(C=C, scale_C=False, **method_kwargs)
                raise NotImplementedError(
                    'save ktrn, multiply Xtst by X in predict()')
                ktrn = linear_kernel(X, X)
                svm.fit(ktrn, y)

            else:
                raise ValueError('unrecognized method', method)

        else:  # n_classes > 2
            if set(y) != set(range(len(set(y)))):
                # TODO: use the label_dct to automatically adjust
                raise NotImplementedError('labels need adapting',
                        set(y))
            if solver == 'auto':
                solver = ('asgd.SparseUpdateRankASGD', {
                        'sgd_step_size0': 0.01,
                        })

            method, method_kwargs = solver

            if method == 'asgd.NaiveRankASGD':
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                auto_step_size0 = method_kwargs.pop('auto_step_size0', True)
                auto_max_examples = method_kwargs.pop('auto_max_examples', 1000)
                svm = NaiveRankASGD(n_classes, n_features,
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                if auto_step_size0:
                    svm = binary_fit(svm, (X, y),
                            max_examples=auto_max_examples)
                else:
                    svm.fit(X, y)

            elif method == 'asgd.SparseUpdateRankASGD':
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                auto_step_size0 = method_kwargs.pop('auto_step_size0', True)
                svm = SparseUpdateRankASGD(n_classes, n_features,
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                if auto_step_size0:
                    svm = binary_fit(svm, (X, y))
                else:
                    svm.fit(X, y)

            elif method in ('asgd.NaiveOVAASGD', 'asgd.OneVsAllASGD'):
                # -- one vs. all
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                auto_step_size0 = method_kwargs.pop(
                        'auto_step_size0', True)
                auto_max_examples = method_kwargs.pop(
                        'auto_max_examples', 1000)
                auto_step_size0_floor = method_kwargs.pop(
                        'auto_step_size0_floor', 1e-7)

                svm = classifier(n_classes, n_features,
                        dtype=method_kwargs.get('dtype', 'float32'))

                if auto_step_size0:
                    all_idxs = method_kwargs['rstate'].permutation(len(X))
                    idxs = all_idxs[:auto_max_examples]
                    step_size0 = find_sgd_step_size0(OneVsAllASGD,
                            svm=svm,
                            data=(X[idxs], y[idxs]),
                            l2_regularization=l2_regularization,
                            **method_kwargs)
                    step_size0 = max(step_size0 / 2.0, auto_step_size0_floor)
                    logger.info('setting sgd_step_size: %e' % step_size0)
                    method_kwargs['sgd_step_size0'] = float(step_size0)

                trainer = OneVsAllASGD(svm, (X, y),
                        l2_regularization=l2_regularization,
                        **method_kwargs)

                for svm in trainer:
                    # this loop does the training
                    pass

            elif method in ('asgd.TheanoOVA',):
                dtype = method_kwargs.get('dtype', X.dtype)
                svm = classifier(n_classes, n_features, dtype=dtype)
                svm = TheanoOVA(svm, data=(X, y),
                        l2_regularization=l2_regularization,
                        **method_kwargs)

            elif method == 'sklearn.svm.SVC':
                # -- one vs. one
                raise NotImplementedError(method)

            elif method == 'sklearn.svm.NuSVC':
                # -- one vs. one
                raise NotImplementedError(method)

            elif method == 'sklearn.svm.LinearSVC':
                # -- Crammer & Singer multi-class
                C = 1.0 / (l2_regularization * len(X))
                svm = sklearn.svm.LinearSVC(
                        C=C,
                        scale_C=False,
                        multi_class=True,
                        **method_kwargs)
                svm.fit(X, y)

            else:
                raise ValueError('unrecognized method', method)

        self.svm = svm

    def predict(self, *args, **kwargs):
        return self.svm.predict(*args, **kwargs)

    def decision_function(self, *args, **kwargs):
        return self.svm.decision_function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


