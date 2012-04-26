import copy
from itertools import izip

import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

import theano
import theano.ifelse
import theano.tensor as tensor

from .naive_asgd import (
        DEFAULT_SGD_STEP_SIZE0,
        DEFAULT_L2_REGULARIZATION,
        DEFAULT_FEEDBACK,
        DEFAULT_RSTATE,
        DEFAULT_DTYPE,
        DEFAULT_SGD_EXPONENT,
        DEFAULT_SGD_TIMESCALE)

from .base import classifier_from_weights

class TheanoBinaryASGD(object):

    """
    Notes regarding speed:
    1. This could be sped up futher by implementing an sdot Op in e.g.
       theano's blas_c.py file.  (currently _dot22 is used for inner product)

    2. Algorithmically, http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
    describes a change of variables that can potentially reduce the number of
    updates required by the algorithm, but I think it returns the same
    advantage as loop fusion / proper BLAS usage.
    """

    sgd_weights = property(
            lambda self: self.s_sgd_weights.get_value(),
            lambda self, val: self.s_sgd_weights.set_value(val))

    sgd_bias = property(
            lambda self: self.s_sgd_bias.get_value(),
            lambda self, val: self.s_sgd_bias.set_value(np.asarray(val)))

    asgd_weights = property(
            lambda self: self.s_asgd_weights.get_value(),
            lambda self, val: self.s_asgd_weights.set_value(val))

    asgd_bias = property(
            lambda self: self.s_asgd_bias.get_value(),
            lambda self, val: self.s_asgd_bias.set_value(np.asarray(val)))

    n_observations = property(
            lambda self: self.s_n_observations.get_value(),
            lambda self, val: self.s_n_observations.set_value(np.asarray(val)))

    sgd_step_size0 = property(
            lambda self: self.s_sgd_step_size0.get_value(),
            lambda self, val: self.s_sgd_step_size0.set_value(np.asarray(val)))

    use_switch = False

    def __init__(self, n_features,
            sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            feedback=DEFAULT_FEEDBACK,
            rstate=DEFAULT_RSTATE,
            dtype=DEFAULT_DTYPE,
            sgd_step_size_scheduling_exponent=DEFAULT_SGD_EXPONENT,
            sgd_step_size_scheduling_multiplier=DEFAULT_SGD_TIMESCALE):

        self.s_n_observations = theano.shared(
                np.asarray(0).astype('int64'),
                name='n_observations',
                allow_downcast=True)

        self.s_sgd_step_size0 = theano.shared(
                np.asarray(0).astype(dtype),
                name='sgd_step_size0',
                allow_downcast=True)

        self.s_sgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='sgd_weights',
                allow_downcast=True)

        self.s_sgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='sgd_bias')

        self.s_asgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='asgd_weights')

        self.s_asgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='asgd_bias')

        BaseASGD.__init__(self,
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            sgd_step_size_scheduling_exponent=sgd_step_size_scheduling_exponent,
            sgd_step_size_scheduling_multiplier=sgd_step_size_scheduling_multiplier)

    def __getstate__(self):
        dct = dict(self.__dict__)
        dynamic_attrs = [
                '_train_fn_2',
                '_tf2_obs',
                '_tf2_idx',
                '_tf2_idxmap',
                '_tf2_mean_cost']
        for attr in dynamic_attrs:
            if attr in dct:
                del dct[attr]
        return dct

    def vector_updates(self, obs, label):
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        l2_regularization = self.l2_regularization

        n_observations = self.s_n_observations
        sgd_step_size0 = self.s_sgd_step_size0

        sgd_weights = self.s_sgd_weights
        sgd_bias = self.s_sgd_bias
        sgd_n = (1 +
                sgd_step_size0 * n_observations
                * sgd_step_size_scheduling_multiplier)
        sgd_step_size = tensor.cast(
                (sgd_step_size0
                    / (sgd_n ** sgd_step_size_scheduling_exponent)),
                sgd_weights.dtype)

        asgd_weights = self.s_asgd_weights
        asgd_bias = self.s_asgd_bias

        # switch to geometric moving average after a while
        # ALSO - this means that mul rather than div is used in the fused
        # elementwise loop that updates asgd_weights, which is faster
        asgd_step_size = tensor.maximum(
                tensor.cast(
                    1.0 / (n_observations + 1),
                    asgd_bias.dtype),
                1e-5)


        margin = label * (tensor.dot(obs, sgd_weights) + sgd_bias)
        regularized_sgd_weights = sgd_weights * tensor.cast(
                1 - l2_regularization * sgd_step_size,
                sgd_weights.dtype)

        if self.use_switch:
            switch = tensor.switch
        else:
            # this is slower to evaluate, but if the features are long and the
            # expected training classification rate is very good, then it
            # can be faster.
            switch = theano.ifelse.ifelse

        assert regularized_sgd_weights.dtype == sgd_weights.dtype
        assert obs.dtype == sgd_weights.dtype
        assert label.dtype == sgd_weights.dtype
        assert sgd_step_size.dtype == sgd_weights.dtype

        new_sgd_weights = switch(margin < 1,
                regularized_sgd_weights + sgd_step_size * label * obs,
                regularized_sgd_weights)
        new_sgd_bias = switch(margin < 1,
                sgd_bias + sgd_step_size * label,
                1 * sgd_bias)
        cost = switch(margin < 1, 1 - margin, 0 * margin)

        new_asgd_weights = ((1.0 - asgd_step_size) * asgd_weights
            + asgd_step_size * new_sgd_weights)
        new_asgd_bias = ((1.0 - asgd_step_size) * asgd_bias
                + asgd_step_size * new_sgd_bias)

        new_n_observations = n_observations + 1

        updates = {
                    sgd_weights: new_sgd_weights,
                    sgd_bias: new_sgd_bias,
                    asgd_weights: new_asgd_weights,
                    asgd_bias: new_asgd_bias,
                    n_observations: new_n_observations}
        return updates, cost

    def compile_train_fn_2(self):
        # This calling strategy is fast, but depends on the use of the CVM
        # linker.
        self._tf2_obs = obs = theano.shared(np.zeros((2, 2),
            dtype=self.dtype),
            allow_downcast=True,
            name='obs')
        # N.B. labels are float
        self._tf2_label = label = theano.shared(np.zeros(2, dtype=self.dtype),
                allow_downcast=True,
                name='label')
        self._tf2_idx = idx = theano.shared(np.asarray(0, dtype='int64'))
        self._tf2_idxmap = idxmap = theano.shared(np.zeros(2, dtype='int64'),
                strict=True)
        self._tf2_mean_cost = mean_cost = theano.shared(
                np.asarray(0, dtype='float64'))
        updates, cost = self.vector_updates(obs[idxmap[idx]], label[idxmap[idx]])
        updates[idx] = idx + 1
        aa = tensor.cast(1.0 / (idx + 1), 'float64')
        # mean cost over idxmap
        updates[mean_cost] = (1 - aa) * mean_cost + aa * cost

        self._train_fn_2 = theano.function([], [],
                updates=updates,
                mode=theano.Mode(
                    optimizer='fast_run',
                    linker='cvm_nogc'))

    def partial_fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()
        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert (n_points,) == y.shape
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        self._tf2_obs.set_value(X, borrow=True)
        # This may cast `y` to a floating point type
        self._tf2_label.set_value(y, borrow=True)
        self._tf2_idxmap.set_value(np.arange(n_points), borrow=True)
        self._tf2_idx.set_value(0)

        if self._train_fn_2.profile:
            for i in xrange(n_points): self._train_fn_2()
        else:
            fn = self._train_fn_2.fn
            for i in xrange(n_points): fn()

        self.train_means.append(self._tf2_mean_cost.get_value()
                    + self.l2_regularization * (self.asgd_weights ** 2).sum())
        return self

    def fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()

        if self.sgd_step_size0 is None:
            self.determine_sgd_step_size0(X, y)

        assert X.ndim == 2
        assert y.ndim == 1
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations
        train_means = self.train_means

        self._tf2_obs.set_value(X, borrow=True)
        self._tf2_label.set_value(y, borrow=True)
        fn = self._train_fn_2.fn

        for i in xrange(n_iterations):
            self._tf2_idxmap.set_value(self.rstate.permutation(n_points))
            self._tf2_idx.set_value(0)
            if self._train_fn_2.profile:
                for i in xrange(n_points): self._train_fn_2()
            else:
                for i in xrange(n_points): fn()
            train_means.append(self._tf2_mean_cost.get_value()
                    + self.l2_regularization * (self.asgd_weights ** 2).sum())
            if self.fit_converged():
                break

        return self

    def decision_function(self, X):
        return (np.dot(self.s_asgd_weights.get_value(borrow=True), X.T)
                + self.asgd_bias)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def reset(self):
        BaseASGD.reset(self)
        self.asgd_weights = self.asgd_weights * 0
        self.asgd_bias = self.asgd_bias * 0
        self.sgd_weights = self.sgd_weights * 0
        self.sgd_bias = self.sgd_bias * 0



def BlockedTheanoOVA(svm, data,
        l2_regularization=1e-3,
        dtype='float64',
        GPU_blocksize=1000 * (1024 ** 2), # bytes
        verbose=False,
        ):
    n_features, n_classes  = svm.weights.shape

    _X = theano.shared(np.ones((2, 2), dtype=dtype),
            allow_downcast=True)
    _yvecs = theano.shared(np.ones((2, 2), dtype=dtype),
            allow_downcast=True)

    sgd_params = tensor.vector(dtype=dtype)

    flat_sgd_weights = sgd_params[:n_features * n_classes]
    sgd_weights = flat_sgd_weights.reshape((n_features, n_classes))
    sgd_bias = sgd_params[n_features * n_classes:]

    margin = _yvecs * (tensor.dot(_X, sgd_weights) + sgd_bias)
    losses = tensor.maximum(0, 1 - margin) ** 2
    l2_cost = .5 * l2_regularization * tensor.dot(
            flat_sgd_weights, flat_sgd_weights)

    cost = losses.mean(axis=0).sum() + l2_cost
    dcost_dparams = tensor.grad(cost, sgd_params)

    _f_df = theano.function([sgd_params], [cost, dcost_dparams])

    assert dtype == 'float32'
    sizeof_dtype = 4
    X, y = data
    yvecs = np.asarray(
            (y[:, None] == np.arange(n_classes)) * 2 - 1,
            dtype=dtype)

    X_blocks = np.ceil(X.size * sizeof_dtype / float(GPU_blocksize))

    examples_per_block = len(X) // X_blocks

    if verbose:
        print 'dividing into', X_blocks, 'blocks of', examples_per_block

    # -- create a dummy class because a nested function cannot modify
    #    params_mean in enclosing scope
    class Dummy(object):
        def __init__(self, collect_estimates):
            params = np.zeros(n_features * n_classes + n_classes)
            params[:n_features * n_classes] = svm.weights.flatten()
            params[n_features * n_classes:] = svm.bias

            self.params = params
            self.params_mean = params.copy().astype('float64')
            self.params_mean_i = 0
            self.collect_estimates = collect_estimates

        def update_mean(self, p):
            self.params_mean_i += 1
            alpha = 1.0 / self.params_mean_i
            self.params_mean *= 1 - alpha
            self.params_mean += alpha * p

        def __call__(self, p):
            if self.collect_estimates:
                self.update_mean(p)
            c, d = _f_df(p.astype(dtype))
            return c.astype('float64'), d.astype('float64')
    dummy = Dummy(X_blocks > 2)

    i = 0
    while i + examples_per_block <= len(X):
        if verbose:
            print 'training on examples', i, 'to', i + examples_per_block
        _X.set_value(
                X[i:i + examples_per_block],
                borrow=True)
        _yvecs.set_value(
                yvecs[i:i + examples_per_block],
                borrow=True)

        best, bestval, info_dct  = fmin_l_bfgs_b(dummy,
                dummy.params_mean.copy(),
                iprint=1 if verbose else -1,
                factr=1e11,  # -- 1e12 for low acc, 1e7 for moderate
                maxfun=1000,
                )
        dummy.update_mean(best)

        i += examples_per_block

    params = dummy.params_mean

    rval = classifier_from_weights(
            weights=params[:n_classes * n_features].reshape(
                (n_features, n_classes)),
            bias=params[n_classes * n_features:])

    return rval


def SubsampledTheanoOVA(svm, data,
        l2_regularization=1e-3,
        dtype='float64',
        feature_bytes=1000 * (1024 ** 2), # bytes
        verbose=False,
        rng=None,
        n_runs=None,  # None -> smallest int that uses all data
        cost_fn='L2Huber',
        bfgs_factr=1e11,  # 1e7 for moderate tolerance, 1e12 for low
        bfgs_maxfun=1000,
        ):
    # I tried to change the problem to work with reduced regularization
    # or a smaller minimal margin (e.g. < 1) to compensate for the missing
    # features, but nothing really worked.
    #
    # I think the better thing would be to do boosting, in just the way we
    # did in the eccv12 project (see e.g. MarginASGD)
    n_features, n_classes = svm.weights.shape
    X, y = data
    if n_runs is None:
        sizeof_dtype = {'float32': 4, 'float64': 8}[dtype]
        Xbytes = X.size * sizeof_dtype
        keep_ratio = float(feature_bytes) / Xbytes
        n_runs = int(np.ceil(1. / keep_ratio))
    n_keep = int(np.ceil(X.shape[1] / float(n_runs)))

    _X = theano.shared(np.ones((2, 2), dtype=dtype),
            allow_downcast=True)
    _yvecs = theano.shared(np.ones((2, 2), dtype=dtype),
            allow_downcast=True)

    sgd_params = tensor.vector(dtype=dtype)
    s_n_use = tensor.lscalar()

    flat_sgd_weights = sgd_params[:s_n_use * n_classes]
    sgd_weights = flat_sgd_weights.reshape((s_n_use, n_classes))
    sgd_bias = sgd_params[s_n_use * n_classes:]

    margin = _yvecs * (tensor.dot(_X, sgd_weights) + sgd_bias)

    if cost_fn == 'L2Half':
        losses = tensor.maximum(0, 1 - margin) ** 2
    elif cost_fn == 'L2Huber':
        # "Huber-ized" L2-SVM
        losses = tensor.switch(
                margin > -1,
                # -- smooth part
                tensor.maximum(0, 1 - margin) ** 2,
                # -- straight part
                -4 * margin)
    elif cost_fn == 'Hinge':
        losses = tensor.maximum(0, 1 - margin)
    else:
        raise ValueError('invalid cost-fn', cost_fn)

    l2_cost = .5 * l2_regularization * tensor.dot(
            flat_sgd_weights, flat_sgd_weights)

    cost = losses.mean(axis=0).sum() + l2_cost
    dcost_dparams = tensor.grad(cost, sgd_params)

    _f_df = theano.function([sgd_params, s_n_use], [cost, dcost_dparams])

    yvecs = np.asarray(
            (y[:, None] == np.arange(n_classes)) * 2 - 1,
            dtype=dtype)

    def flatten_svm(obj):
        return np.concatenate([obj.weights.flatten(), obj.bias])

    if verbose:
        print 'keeping', n_keep, 'of', X.shape[1], 'features'

    if rng is None:
        rng = np.random.RandomState(123)

    all_feat_randomized = rng.permutation(X.shape[1])
    bests = []
    for ii in range(n_runs):
        use_features = all_feat_randomized[ii * n_keep: (ii + 1) * n_keep]
        assert len(use_features)
        n_use = len(use_features)

        def f(p):
            c, d = _f_df(p.astype(dtype), n_use)
            return c.astype('float64'), d.astype('float64')

        params = np.zeros(n_use * n_classes + n_classes)
        params[:n_use * n_classes] = svm.weights[use_features].flatten()
        params[n_use * n_classes:] = svm.bias

        _X.set_value(X[:, use_features], borrow=True)
        _yvecs.set_value(yvecs, borrow=True)

        best, bestval, info_dct = fmin_l_bfgs_b(f,
                params,
                iprint=1 if verbose else -1,
                factr=bfgs_factr,  # -- 1e12 for low acc, 1e7 for moderate
                maxfun=bfgs_maxfun,
                )
        best_svm = copy.deepcopy(svm)
        best_svm.weights[use_features] = best[:n_classes * n_use].reshape(
                    (n_use, n_classes))
        best_svm.bias = best[n_classes * n_use:]
        bests.append(flatten_svm(best_svm))

    # sum instead of mean here, because each loop iter trains only a subset of
    # features. XXX: This assumes that those subsets are mutually exclusive
    best_params = np.sum(bests, axis=0)
    rval = copy.deepcopy(svm)
    rval.weights = best_params[:n_classes * n_features].reshape(
                (n_features, n_classes))
    rval.bias = best_params[n_classes * n_features:]
    return rval


# XXX REFACTOR WITH SubsampledTheanoOVA
def BinarySubsampledTheanoOVA(svm, data,
        l2_regularization=1e-3,
        dtype='float64',
        feature_bytes=1000 * (1024 ** 2), # bytes
        verbose=False,
        rng=None,
        n_runs=None,  # None -> smallest int that uses all data
        cost_fn='L2Huber',
        bfgs_factr=1e11,  # 1e7 for moderate tolerance, 1e12 for low
        bfgs_maxfun=1000,
        decisions=None
        ):
    n_features, = svm.weights.shape
    X, y = data

    # XXX REFACTOR
    if n_runs is None:
        sizeof_dtype = {'float32': 4, 'float64': 8}[dtype]
        Xbytes = X.size * sizeof_dtype
        keep_ratio = float(feature_bytes) / Xbytes
        n_runs = int(np.ceil(1. / keep_ratio))
        print 'BinarySubsampledTheanoOVA using n_runs =', n_runs
    n_keep = int(np.ceil(X.shape[1] / float(n_runs)))

    assert set(y) == set([-1, 1])
    _X = theano.shared(np.ones((2, 2), dtype=dtype),
            allow_downcast=True)
    _yvecs = theano.shared(y.astype(dtype),
            allow_downcast=True)
    if decisions:
        decisions = np.asarray(decisions).astype(dtype)
        # -- N.B. for multi-class the decisions would be an examples x classes
        # matrix
        if decisions.shape != y.shape:
            raise ValueError('decisions have wrong shape', decisions.shape)
        _decisions = theano.shared(decisions)
        del decisions
    else:
        _decisions = theano.shared(y.astype(dtype) * 0, allow_downcast=True)

    sgd_params = tensor.vector(dtype=dtype)
    s_n_use = tensor.lscalar()

    sgd_weights = sgd_params[:s_n_use]
    sgd_bias = sgd_params[s_n_use]

    margin = _yvecs * (tensor.dot(_X, sgd_weights) + sgd_bias + _decisions)

    # XXX REFACTOR
    if cost_fn == 'L2Half':
        losses = tensor.maximum(0, 1 - margin) ** 2
    elif cost_fn == 'L2Huber':
        # "Huber-ized" L2-SVM
        losses = tensor.switch(
                margin > -1,
                # -- smooth part
                tensor.maximum(0, 1 - margin) ** 2,
                # -- straight part
                -4 * margin)
    elif cost_fn == 'Hinge':
        losses = tensor.maximum(0, 1 - margin)
    else:
        raise ValueError('invalid cost-fn', cost_fn)

    l2_cost = .5 * l2_regularization * tensor.dot(
            sgd_weights, sgd_weights)

    cost = losses.mean() + l2_cost
    dcost_dparams = tensor.grad(cost, sgd_params)

    _f_df = theano.function([sgd_params, s_n_use], [cost, dcost_dparams])

    _f_update_decisions = theano.function([sgd_params, s_n_use], [],
            updates={
                _decisions: (
                    tensor.dot(_X, sgd_weights) + sgd_bias + _decisions),
                })

    def flatten_svm(obj):
        # Note this is different from multi-class case because bias is scalar
        return np.concatenate([obj.weights.flatten(), [obj.bias]])

    if verbose:
        print 'keeping', n_keep, 'of', X.shape[1], 'features, per round'
        print 'running for ', n_runs, 'rounds'

    if rng is None:
        rng = np.random.RandomState(123)

    all_feat_randomized = rng.permutation(X.shape[1])
    bests = []
    for ii in range(n_runs):
        use_features = all_feat_randomized[ii * n_keep: (ii + 1) * n_keep]
        assert len(use_features)
        n_use = len(use_features)

        def f(p):
            c, d = _f_df(p.astype(dtype), n_use)
            return c.astype('float64'), d.astype('float64')

        params = np.zeros(n_use + 1)
        params[:n_use] = svm.weights[use_features].flatten()
        params[n_use] = svm.bias

        _X.set_value(X[:, use_features], borrow=True)

        best, bestval, info_dct = fmin_l_bfgs_b(f,
                params,
                iprint=int(verbose) - 1,
                factr=bfgs_factr,  # -- 1e12 for low acc, 1e7 for moderate
                maxfun=bfgs_maxfun,
                )
        _f_update_decisions(best.astype(dtype), n_use)
        best_svm = copy.deepcopy(svm)
        best_svm.weights[use_features] = best[:n_use].astype(dtype)
        best_svm.bias = float(best[n_use])
        bests.append(flatten_svm(best_svm))

    best_params = np.sum(bests, axis=0)
    rval = copy.deepcopy(svm)
    rval.weights = best_params[:n_features].astype(dtype)
    rval.bias = float(best_params[n_features])

    # XXX: figure out why Theano may be not freeing this memory?
    _X.set_value(np.ones((2, 2), dtype=dtype))
    _yvecs.set_value(np.ones(2, dtype=dtype))
    return rval
