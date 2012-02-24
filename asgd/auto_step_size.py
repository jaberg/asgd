import copy
import logging
import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_RANGE = 0.25, 0.5
DEFAULT_MAX_EXAMPLES = 1000  # estimate stepsize from this many examples
DEFAULT_TOLERANCE = 0.01     # in logarithmic units of the training criterion
DEFAULT_BRENT_OUTPUT = False



def find_sgd_step_size0(
    model, partial_fit_args,
    initial_range=DEFAULT_INITIAL_RANGE,
    tolerance=DEFAULT_TOLERANCE, brent_output=DEFAULT_BRENT_OUTPUT):
    """Use a Brent line search to find the best step size

    Parameters
    ----------
    model: BinaryASGD
        Instance of a BinaryASGD

    partial_fit_args - tuple of arguments for model.partial_fit.
        This tuple must start with X, y, ...

    initial_range: tuple of float
        Initial range for the sgd_step_size0 search (low, high)

    max_iterations:
        Maximum number of interations

    Returns
    -------
    best_sgd_step_size0: float
        Optimal sgd_step_size0 given `X` and `y`.
    """
    # -- stupid scipy calls some sizes twice!?
    _cache = {}

    def eval_size0(log2_size0):
        try:
            return _cache[log2_size0]
        except KeyError:
            pass
        other = copy.deepcopy(model)
        current_step_size = 2 ** log2_size0
        other.sgd_step_size0 = current_step_size
        other.sgd_step_size = current_step_size
        other.partial_fit(*partial_fit_args)
        # Hack: asgd is lower variance than sgd, but it's tuned to work
        # well asymptotically, not after just a few examples
        weights = .5 * (other.asgd_weights + other.sgd_weights)
        bias = .5 * (other.asgd_bias + other.sgd_bias)

        X, y = partial_fit_args[:2]
        margin = y * (np.dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        rval = np.maximum(0, 1 - margin).mean() + l2_cost
        if np.isnan(rval):
            rval = float('inf')
        # -- apply minimizer in log domain
        rval = np.log(rval)
        _cache[log2_size0] = rval
        return rval

    log2_best_sgd_step_size0 = optimize.brent(
        eval_size0, brack=np.log2(initial_range), tol=tolerance)

    rval = max(2 ** log2_best_sgd_step_size0, 1e-7)
    return rval


def binary_fit(
    model, fit_args,
    max_examples=DEFAULT_MAX_EXAMPLES,
    **find_sgd_step_size0_kwargs):
    """Returns a model with automatically-selected sgd_step_size0

    Parameters
    ----------
    model: BinaryASGD
        Instance of the model to be fitted.

    fit_args - tuple of args to model.fit
        This method assumes they are all length-of-dataset ndarrays.

    max_examples: int
        Maximum number of examples to use from `X` and `y` to find an
        estimate of the best sgd_step_size0. N.B. That the entirety of X and y
        is used for the final fit() call after the best step size has been found.

    Returns
    -------
    model: BinaryASGD
        Instances of the model, fitted with an estimate of the best
        sgd_step_size0
    """

    assert max_examples > 0

    # randomly choose up to max_examples uniformly without replacement from
    # across the whole set of training data.
    all_idxs = model.rstate.permutation(len(fit_args[0]))
    idxs = all_idxs[:max_examples]

    # Find the best learning rate for that subset
    best = find_sgd_step_size0(
        model, [a[idxs] for a in fit_args], **find_sgd_step_size0_kwargs)
    logger.info('found best: %f' % best)

    # Heuristic: take the best stepsize according to the first max_examples,
    # and go half that fast for the full run.
    stepdown = 5 * np.sqrt( float(len(all_idxs)) / float(len(idxs)))

    logger.info('setting sgd_step_size: %f' % (best/stepdown))
    model.sgd_step_size0 = best / stepdown
    model.sgd_step_size = best / stepdown
    model.fit(*fit_args)

    return model
