import sys
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


def test_error_rate(y, yhat, return_stderr=False):
    mu = np.mean((y != yhat).astype('float'))
    var = mu * (1 - mu) / (len(y) - 1)
    stderr = np.sqrt(var)
    if return_stderr:
        return mu, stderr
    else:
        return mu


DOT_MAX_NDIMS = 256
MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000

if 0:

    import theano
    sX = theano.tensor.matrix(dtype='float32')
    sY = theano.tensor.matrix(dtype='float32')
    dot32 = theano.function([sX, sY], theano.tensor.dot(sX, sY))
    sX = theano.tensor.matrix(dtype='float64')
    sY = theano.tensor.matrix(dtype='float64')
    dot64 = theano.function([sX, sY], theano.tensor.dot(sX, sY))
    def dot(A, B):
        _dot = dict(float32=dot32, float64=dot64)[str(A.dtype)]
        return _dot(A, B)

else:
    dot = np.dot


def chunked_linear_kernel(Xs, Ys, symmetric):
    """Compute a linear kernel in blocks so that it can use a GPU with limited
    memory.

    Xs is a list of feature matrices
    Ys ia  list of feature matrices

    This function computes the kernel matrix with
        \sum_i len(Xs[i]) rows
        \sum_j len(Ys[j]) cols
    """

    dtype = Xs[0].dtype

    def _dot(A, B):
        if K < DOT_MAX_NDIMS:
            return dot(A, B)
        else:
            out = dot(A[:,:DOT_MAX_NDIMS], B[:DOT_MAX_NDIMS])
            ndims_done = DOT_MAX_NDIMS            
            while ndims_done < K:
                out += dot(
                    A[:,ndims_done : ndims_done + DOT_MAX_NDIMS], 
                    B[ndims_done : ndims_done + DOT_MAX_NDIMS])
                ndims_done += DOT_MAX_NDIMS
            return out

    R = sum([len(X) for X in Xs])
    C = sum([len(Y) for Y in Ys])
    K = Xs[0].shape[1]

    rval = np.zeros((R, C), dtype=dtype)

    if symmetric:
        assert R == C

    print 'Computing gram matrix',

    ii0 = 0
    for ii, X_i in enumerate(Xs):
        sys.stdout.write('.')
        sys.stdout.flush()
        ii1 = ii0 + len(X_i) # -- upper bound of X block

        jj0 = 0
        for jj, Y_j in enumerate(Ys):
            jj1 = jj0 + len(Y_j) # -- upper bound of Y block

            r_ij = rval[ii0:ii1, jj0:jj1]

            if symmetric and jj < ii:
                r_ji = rval[jj0:jj1, ii0:ii1]
                r_ij[:] = r_ji.T
            else:
                r_ij[:] = _dot(X_i, Y_j.T)

            jj0 = jj1

        ii0 = ii1

    print 'done!'
    return rval


def linear_kernel(X, Y, block_size=10000):
    """Compute a linear kernel in blocks so that it can use a GPU with limited
    memory.

    Xs is a list of feature matrices
    Ys ia  list of feature matrices

    This function computes the kernel matrix with
        \sum_i len(Xs[i]) rows
        \sum_j len(Ys[j]) cols
    """

    def chunk(Z):
        Zs = []
        ii = 0
        while len(Z[ii:ii + block_size]):
            Zs.append(Z[ii:ii + block_size])
            ii += block_size
        return Zs

    Xs = chunk(X)
    Ys = chunk(Y)

    assert sum([len(xi) for xi in Xs]) == len(X)
    assert sum([len(yi) for yi in Ys]) == len(Y)
    return chunked_linear_kernel(Xs, Ys, symmetric=(X is Y))
