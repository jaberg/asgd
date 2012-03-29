
import numpy as np

def mean_and_std(X, min_std):
    # XXX: REPLACE THIS WITH eccv12/utils.mean_and_std
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-8,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    train_mean, train_std = mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


def geometric_bracket_min(f, x0, x1, f_thresh):
    """
    Find a pair (x2, x3) whose ratio x2/x3 == x0/x1, that bracket x*
    minimizing f
    """
    if x0 >= x1:
        raise ValueError('x0 must be < x1', (x0, x1))
    factor = x1 / x0
    y0 = float(f(x0))
    y1 = float(f(x1))
    if y0 < y1:
        while y0 + f_thresh < y1:
            x1 = x0
            y1 = y0
            x0 = x1 / factor
            y0 = float(f(x0))
    elif y1 < y0:
        while y1 + f_thresh < y0:
            x0 = x1
            y0 = y1
            x1 = x0 * factor
            y1 = float(f(x1))
    return (x0, x1)

