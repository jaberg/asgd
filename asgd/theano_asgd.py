
import numpy as np
from itertools import izip

from naive_asgd import NaiveBinaryASGD

from naive_asgd import (
        DEFAULT_SGD_STEP_SIZE0,
        DEFAULT_L2_REGULARIZATION,
        DEFAULT_N_ITERATIONS,
        DEFAULT_FEEDBACK,
        DEFAULT_RSTATE,
        DEFAULT_DTYPE,
        DEFAULT_SGD_EXPONENT,
        DEFAULT_SGD_TIMESCALE)

import theano
import theano.ifelse
import theano.tensor as tensor


def vector_updates(
        # symbolic args
        obs,
        label,
        n_observations,
        sgd_step_size0,
        sgd_weights,
        sgd_bias,
        asgd_weights,
        asgd_bias,
        # non-symbolic args
        l2_regularization,
        sgd_step_size_scheduling_exponent,
        sgd_step_size_scheduling_multiplier,
        use_switch,
        ):

    sgd_n = (1 +
            sgd_step_size0 * n_observations
            * sgd_step_size_scheduling_multiplier)
    sgd_step_size = tensor.cast(
            (sgd_step_size0
                / (sgd_n ** sgd_step_size_scheduling_exponent)),
            sgd_weights.dtype)

    # switch to geometric moving average after a while
    # ALSO - this means that mul rather than div is used in the fused
    # elementwise loop that updates asgd_weights, which is faster
    asgd_step_size = tensor.maximum(
            tensor.cast(
                1.0 / (n_observations + 1),
                asgd_bias.dtype),
            1e-5)

    if use_switch:
        switch = theano.tensor.switch
    else:
        switch = theano.lazy.ifelse

    margin = label * (tensor.dot(obs, sgd_weights) + sgd_bias)
    regularized_sgd_weights = sgd_weights * tensor.cast(
            1 - l2_regularization * sgd_step_size,
            sgd_weights.dtype)

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

_fn_cache = {}
# some vars could be a svar element rather than signature constant:
#  - step_size exponent
#  - step_size_multiplier
#  - l2_regularization
def get_fit_fn(
        dtype,
        l2_regularization,
        sgd_step_size_scheduling_exponent,
        sgd_step_size_scheduling_multiplier,
        use_switch,
        ):
    # This calling strategy is fast, but depends on the use of the CVM
    # linker.
    dtype = str(dtype)

    signature = (
            dtype,
            l2_regularization,
            sgd_step_size_scheduling_exponent,
            sgd_step_size_scheduling_multiplier,
            use_switch)
    try:
        return _fn_cache[signature]
    except KeyError:
        pass

    svar = {}
    svar['n_observations'] = theano.shared(
                np.asarray(0).astype('int64'),
                name='n_observations',
                allow_downcast=True)

    svar['sgd_step_size0'] = theano.shared(
                np.asarray(0).astype(dtype),
                name='sgd_step_size0',
                allow_downcast=True)

    svar['sgd_weights'] = theano.shared(
                np.zeros(2, dtype=dtype),
                name='sgd_weights',
                allow_downcast=True)

    svar['sgd_bias'] = theano.shared(
                np.asarray(0, dtype=dtype),
                name='sgd_bias')

    svar['asgd_weights'] = theano.shared(
                np.zeros(2, dtype=dtype),
                name='asgd_weights')

    svar['asgd_bias'] = theano.shared(
                np.asarray(0, dtype=dtype),
                name='asgd_bias')

    svar['obs'] = obs = theano.shared(np.zeros((2, 2),
            dtype=dtype),
            allow_downcast=True,
            name='obs')

    # N.B. labels are float
    svar['label'] = label = theano.shared(np.zeros(2, dtype=dtype),
            allow_downcast=True,
            name='label')
    svar['idx'] = idx = theano.shared(
            np.asarray(0, dtype='int64'),
            name='idx')
    svar['idxmap'] = idxmap = theano.shared(
            np.zeros(2, dtype='int64'),
            strict=True,
            name='idxmap')
    svar['mean_cost'] = mean_cost = theano.shared(
            np.asarray(0, dtype=dtype),
            name='mean_cost')

    updates, cost = vector_updates(
            obs[idxmap[idx]],
            label[idxmap[idx]],
            svar['n_observations'],
            svar['sgd_step_size0'],
            svar['sgd_weights'],
            svar['sgd_bias'],
            svar['asgd_weights'],
            svar['asgd_bias'],
            l2_regularization,
            sgd_step_size_scheduling_exponent,
            sgd_step_size_scheduling_multiplier,
            use_switch)

    updates[idx] = idx + 1
    # mean cost over idxmap
    aa = tensor.cast(1.0 / (idx + 1), dtype)
    updates[mean_cost] = (1 - aa) * mean_cost + aa * cost

    fn = theano.function([], [],
            updates=updates,
            mode=theano.Mode( optimizer='fast_run', linker='cvm_nogc'),
            )

    _fn_cache[signature] = (updates, cost, svar, fn)

    return _fn_cache[signature]



class TheanoBinaryASGD(NaiveBinaryASGD):

    """
    Notes regarding speed:
    1. This could be sped up futher by implementing an sdot Op in e.g.
       theano's blas_c.py file.  (currently _dot22 is used for inner product)

    2. Algorithmically, http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
    describes a change of variables that can potentially reduce the number of
    updates required by the algorithm, but I think it returns the same
    advantage as loop fusion / proper BLAS usage.
    """

    use_switch = True

    def partial_fit(self, X, y):
        updates, cost, svar, fn = get_fit_fn(
                self.asgd_weights.dtype,
                self.l2_regularization,
                self.sgd_step_size_scheduling_exponent,
                self.sgd_step_size_scheduling_multiplier,
                self.use_switch)
        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert (n_points,) == y.shape
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        params = [
                'sgd_weights',
                'sgd_bias',
                'asgd_weights',
                'asgd_bias',
                'n_observations',
                'sgd_step_size0']

        for key, val in [
                ('obs', X),
                ('label', y),
                ('idx', 0),
                ('idxmap', np.arange(n_points)),
                ]:
            svar[key].set_value(val, borrow=True)

        for key in params:
            svar[key].set_value(getattr(self, key), borrow=True)

        if fn.profile:
            for i in xrange(n_points): fn()
        else:
            fn_fn = fn.fn
            for i in xrange(n_points): fn_fn()

        for key in params:
            setattr(self, key, svar[key].get_value(borrow=False))

        self.train_means.append(
                svar['mean_cost'].get_value()
                + self.l2_regularization * (self.asgd_weights ** 2).sum())

        return self


    def fit(self, X, y):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        n_iterations = self.n_iterations

        if self.sgd_step_size0 is None:
            self.determine_sgd_step_size0(X, y)
        #
        # XXX don't overlap with determine_sgd_step_size0 until the get_fit_fn
        # is reentrant.
        #
        updates, cost, svar, fn = get_fit_fn(
                self.asgd_weights.dtype,
                self.l2_regularization,
                self.sgd_step_size_scheduling_exponent,
                self.sgd_step_size_scheduling_multiplier,
                self.use_switch)

        params = [
                'sgd_weights',
                'sgd_bias',
                'asgd_weights',
                'asgd_bias',
                'n_observations',
                'sgd_step_size0']

        for key, val in [
                ('obs', X),
                ('label', y),
                ('idx', 0),
                ('idxmap', np.arange(n_points)),
                ]:
            svar[key].set_value(val, borrow=True)

        for key in params:
            svar[key].set_value(getattr(self, key), borrow=True)

        for i in xrange(n_iterations):
            svar['idxmap'].set_value(self.rstate.permutation(n_points))
            svar['idx'].set_value(0)

            if fn.profile:
                for i in xrange(n_points): fn()
            else:
                fn_fn = fn.fn
                for i in xrange(n_points): fn_fn()

            self.train_means.append(
                    svar['mean_cost'].get_value()
                    + self.l2_regularization * (self.asgd_weights ** 2).sum())

            if self.fit_converged():
                break

        for key in params:
            setattr(self, key, svar[key].get_value(borrow=False))

        return self

