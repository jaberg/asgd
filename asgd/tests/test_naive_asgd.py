import sys
from nose.tools import assert_equal, raises
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
import numpy as np
from numpy.random import RandomState

from asgd import NaiveBinaryASGD as BinaryASGD
from asgd import NaiveOVAASGD as OVAASGD

RTOL = 1e-6
ATOL = 1e-6

N_POINTS = 1e3
N_FEATURES = 1e2

DEFAULT_ARGS = (N_FEATURES,)
DEFAULT_KWARGS = dict(sgd_step_size0=1e-3,
                      l2_regularization=1e-6,
                      min_observations=4 * N_POINTS,
                      max_observations=4 * N_POINTS,
                      dtype=np.float32)


def get_fake_data(n_points, n_features, rstate):
    X = rstate.randn(n_points, n_features).astype(np.float32)
    y = 2 * (rstate.randn(n_points) > 0) - 1
    X[y == 1] += 1e-1
    return X, y


def get_fake_binary_data_multi_labels(n_points, n_features, rstate):
    X = rstate.randn(n_points, n_features).astype(np.float32)
    y = rstate.randn(n_points) > 0
    X[y] += 1e-1
    return X, y


def get_fake_multiclass_data(n_points, n_features, n_classes, rstate):
    X = rstate.randn(n_points, n_features).astype(np.float32)
    y = rstate.randint(n_classes, size=n_points)
    return X, y


def test_naive_asgd():
    rstate = RandomState(42)

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    clf = BinaryASGD(*DEFAULT_ARGS, rstate=rstate, **DEFAULT_KWARGS)

    clf.fit(X, y)
    assert clf.n_observations == clf.min_observations == clf.max_observations
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.72)
    assert_equal(ytst_acc, 0.522)


def test_naive_asgd_with_feedback():
    raise SkipTest("FIXME: feedback support is buggy")

    rstate = RandomState(43)

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    clf = BinaryASGD(*DEFAULT_ARGS,
                     feedback=True, rstate=rstate, **DEFAULT_KWARGS)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.709)
    assert_equal(ytst_acc, 0.449)


def test_naive_asgd_multi_labels():
    rstate = RandomState(44)

    X, y = get_fake_binary_data_multi_labels(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_binary_data_multi_labels(N_POINTS, N_FEATURES,
                                                   rstate)

    # n_classes is 2 since it is actually a binary case
    clf = OVAASGD(2, N_FEATURES, sgd_step_size0=1e-3, l2_regularization=1e-6,
                  min_observations=4 * N_POINTS,
                  max_observations=4 * N_POINTS,
                  dtype=np.float32,
                  rstate=rstate)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.728)
    assert_equal(ytst_acc, 0.52)


def test_naive_multiclass_ova_asgd():
    rstate = RandomState(45)

    n_classes = 10

    X, y = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes, rstate)
    Xtst, ytst = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes,
                                          rstate)

    clf = OVAASGD(n_classes, N_FEATURES, sgd_step_size0=1e-3,
                  l2_regularization=1e-6,
                  min_observations=4 * N_POINTS,
                  max_observations=4 * N_POINTS,
                  dtype=np.float32,
                  rstate=rstate)

    clf.fit(X, y)

    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()

    assert_equal(ytrn_acc, 0.335)
    assert_equal(ytst_acc, 0.092)


def test_naive_multiclass_ova_vs_binary_asgd():
    rstate = RandomState(42)

    n_classes = 3

    Xtrn, ytrn = get_fake_multiclass_data(
        N_POINTS, N_FEATURES, n_classes, rstate)
    Xtst, ytst = get_fake_multiclass_data(
        N_POINTS, N_FEATURES, n_classes, rstate)

    # -- ground truth 'gt': emulate OVA with binary asgd classifiers
    # binary assignments
    ytrns = ytrn[np.newaxis, :] == np.arange(n_classes)[:, np.newaxis]
    # transform into -1 / +1
    ytrns = 2 * ytrns - 1
    # get individual predictions
    basgds = [
        BinaryASGD(
            *DEFAULT_ARGS,
            rstate=RandomState(999),
            **DEFAULT_KWARGS).partial_fit(Xtrn, ysel)
        for ysel in ytrns
    ]
    ytsts = [basgd.decision_function(Xtst) for basgd in basgds]
    # convert to array of shape (n_classes, n_points)
    gt = np.array(ytsts).T

    # -- given 'gv': use the OVA class
    clf = OVAASGD(*((n_classes,) + DEFAULT_ARGS),
                  rstate=RandomState(999), **DEFAULT_KWARGS)
    clf.partial_fit(Xtrn, ytrn)
    gv = clf.decision_function(Xtst)

    # -- test
    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)


@raises(ValueError)
def test_naive_ova_asgd_wrong_labels():
    rstate = RandomState(42)

    n_classes = 10

    Xtrn, ytrn = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes,
                                          rstate)

    clf = OVAASGD(*((n_classes,) + DEFAULT_ARGS),
                  rstate=RandomState(999), **DEFAULT_KWARGS)
    ytrn_bad = rstate.randint(n_classes + 42, size=len(ytrn))
    clf.partial_fit(Xtrn, ytrn_bad)


def test_early_stopping():
    rstate = RandomState(42)

    for N_POINTS in (1000, 2000):

        X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
        Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

        kwargs = dict(DEFAULT_KWARGS)
        kwargs['fit_n_partial'] = 1500
        del kwargs['max_observations']
        print kwargs
        print 'N_POINTS', N_POINTS

        clf = BinaryASGD(*DEFAULT_ARGS, rstate=rstate, **kwargs)

        clf.fit(X, y)
        print clf.fit_n_partial
        print len(clf.train_means)
        print clf.n_observations
        assert clf.fit_n_partial == kwargs['fit_n_partial']
        assert len(clf.train_means) == clf.n_observations / clf.fit_n_partial
        assert clf.min_observations == kwargs['min_observations']
        assert clf.max_observations == sys.maxint
        assert clf.n_observations >= kwargs['min_observations']
        assert clf.fit_converged()
        ytrn_preds = clf.predict(X)
        ytrn_acc = (ytrn_preds == y).mean()
        assert ytrn_acc > 0.7
