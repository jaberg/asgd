from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from numpy.random import RandomState
from asgd import NaiveBinaryASGD as BinaryASGD
from asgd.auto_step_size import find_sgd_step_size0
from asgd.auto_step_size import binary_fit
from asgd.auto_step_size import DEFAULT_MAX_EXAMPLES

from test_naive_asgd import get_fake_data


def get_new_model(n_features, rstate, n_points):
    return BinaryASGD(n_features, rstate=rstate,
            sgd_step_size0=1e3,  #  intentionally large, binary_fit will fix
            l2_regularization=1e-3,
            max_observations=5 * n_points,
            min_observations=5 * n_points,
            )


def test_binary_sgd_step_size0():
    rstate = RandomState(42)
    n_features = 20

    X, y = get_fake_data(100, n_features, rstate)

    clf = get_new_model(n_features, rstate, 100)
    best0 = find_sgd_step_size0(clf, (X, y))
    print best0
    assert np.allclose(best0, 0.04, atol=.1, rtol=.5)

    # find_sgd_step_size0 does not change clf
    assert clf.sgd_step_size0 == 1000.0


def test_binary_fit():
    rstate = RandomState(42)
    n_features = 20

    for L in [100, DEFAULT_MAX_EXAMPLES, int(DEFAULT_MAX_EXAMPLES * 1.5),
            int(DEFAULT_MAX_EXAMPLES * 3)]:

        clf = get_new_model(n_features, rstate, L)
        X, y = get_fake_data(L, n_features, rstate, separation=0.1)
        best = find_sgd_step_size0(clf, (X, y))
        _clf = binary_fit(clf, (X, y))
        assert _clf is clf
        assert 0 < clf.sgd_step_size0 <= best


def test_fit_replicable():

    n_features = 20

    X, y = get_fake_data(100, n_features, RandomState(4))

    m0 = get_new_model(n_features, RandomState(45), 100)
    m0 = binary_fit(m0, (X, y))

    m1 = get_new_model(n_features, RandomState(45), 100)
    m1 = binary_fit(m1, (X, y))

    assert_array_equal(m0.sgd_weights, m1.sgd_weights)
    assert_array_equal(m0.sgd_bias, m1.sgd_bias)
