import copy
import logging
import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXAMPLES = 1000  # estimate stepsize from this many examples
DEFAULT_TOLERANCE = 1.0     # in log-2 units of the learning rate
DEFAULT_SGD_STEP_SIZE_FLOOR = 1e-7  # -- for huge feature vectors, reduce this.


def find_sgd_step_size0(
    model, partial_fit_args,
    tolerance=DEFAULT_TOLERANCE):
    """Use a Brent line search to find the best step size

    Parameters
    ----------
    model: BinaryASGD
        Instance of a BinaryASGD

    partial_fit_args - tuple of arguments for model.partial_fit.
        This tuple must start with X, y, ...

    tolerance: in logarithmic step size units

    Returns
    -------
        Optimal sgd_step_size0 given `X` and `y`.
    """
    # -- stupid scipy calls some sizes twice!?
    _cache = {}

    def eval_size0(log2_size0):
        try:
            return _cache[float(log2_size0)]
        except KeyError:
            pass
        other = copy.deepcopy(model)
        current_step_size = 2 ** log2_size0
        other.sgd_step_size0 = current_step_size
        other.sgd_step_size = current_step_size
        other.partial_fit(*partial_fit_args)
        # Hack: asgd is lower variance than sgd, but it's tuned to work
        # well asymptotically, not after just a few examples
        other.asgd_weights = .5 * (other.asgd_weights + other.sgd_weights)
        other.asgd_bias = .5 * (other.asgd_bias + other.sgd_bias)

        X, y = partial_fit_args[:2]
        rval = other.cost(X, y)
        if np.isnan(rval):
            rval = float('inf')
        #print 'find step %e: %e' % (current_step_size, rval)
        return rval

    log2_best_sgd_step_size0 = scipy.optimize.fmin(
            eval_size0,
            x0=np.log2(model.sgd_step_size0),
            xtol=tolerance,
            )

    #print 'Brent came back with', log2_best_sgd_step_size0, 2 ** log2_best_sgd_step_size0

    rval = 2.0 ** log2_best_sgd_step_size0
    return rval


# XXX: use different name, function is not specific to binary classification
def binary_fit(
    model, fit_args,
    max_examples=DEFAULT_MAX_EXAMPLES,
    step_size_floor=DEFAULT_SGD_STEP_SIZE_FLOOR,
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
    logger.info('found best: %e' % best)

    # Heuristic: take the best stepsize according to the first max_examples,
    # and go half that fast for the full run.
    stepdown = np.sqrt( float(len(all_idxs)) / float(len(idxs)))
    step_size0 = max(best / stepdown, step_size_floor)

    logger.info('setting sgd_step_size: %e' % step_size0)
    model.sgd_step_size0 = float(step_size0)
    model.sgd_step_size = float(step_size0)
    model.fit(*fit_args)

    return model
