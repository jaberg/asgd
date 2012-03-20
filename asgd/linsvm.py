import numpy as np

from .auto_step_size import binary_fit
from .naive_asgd import NaiveBinaryASGD
from .naive_asgd import NaiveRankASGD
from .naive_asgd import SparseUpdateRankASGD

try:
    import sklearn.svm
except ImportError:
    pass


class LinearSVM(object):
    """
    SVM-fitting object that implements a heuristic for choosing
    the right solver among several that may be installed in sklearn, and asgd.

    """

    def __init__(self, l2_regularization, solver='auto', label_dct=None,
            label_weights=None):
        self.l2_regularization = l2_regularization
        self.solver = solver
        self.label_dct = label_dct
        self.label_weights = label_weights

    def fit(self, X, y, weights=None, bias=None):
        solver = self.solver
        label_dct = self.label_dct
        l2_regularization = self.l2_regularization

        if weights or bias:
            raise NotImplementedError(
                    'Currently only train_set = (X, y) is supported')
        del weights, bias
        if self.label_weights:
            raise NotImplementedError()

        n_train, n_feats = X.shape
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
                if n_feats > n_train:
                    solver = ('sklearn.svm.SVC', {'kernel': 'precomputed'})
                else:
                    solver = ('asgd.NaiveBinaryASGD', {})

            method, method_kwargs = solver

            if method == 'asgd.NaiveBinaryASGD':
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                svm = NaiveBinaryASGD(
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                svm = binary_fit(svm, (X, y))

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
                # TODO: switch to OVA and use libSVM?
                #if n_feats > n_train:
                #solver = ('asgd.NaiveRankASGD', { })
                solver = ('asgd.SparseUpdateRankASGD', {})

            method, method_kwargs = solver

            if method == 'asgd.NaiveRankASGD':
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                svm = NaiveRankASGD(n_classes, n_feats,
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                svm = binary_fit(svm, (X, y))

            elif method == 'asgd.SparseUpdateRankASGD':
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                svm = SparseUpdateRankASGD(n_classes, n_feats,
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                svm = binary_fit(svm, (X, y))

            elif method == 'asgd.NaiveOVAASGD':
                # -- one vs. all
                method_kwargs = dict(method_kwargs)
                method_kwargs.setdefault('rstate', np.random.RandomState(123))
                svm = NaiveOVAASGD(n_classes, n_feats,
                        l2_regularization=l2_regularization,
                        **method_kwargs)
                svm = binary_fit(svm, (X, y))

            elif method == 'sklearn.svm.SVC':
                # -- one vs. one
                raise NotImplementedError(method)

            elif method == 'sklearn.svm.NuSVC':
                # -- one vs. one
                raise NotImplementedError(method)

            elif method == 'sklearn.svm.LinearSVC':
                # -- one vs. all
                raise NotImplementedError(method)

            else:
                raise ValueError('unrecognized method', method)

        self.svm = svm

    def predict(self, *args, **kwargs):
        return self.svm.predict(*args, **kwargs)

    def decision_function(self, *args, **kwargs):
        return self.svm.decision_function(*args, **kwargs)


