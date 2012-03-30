import numpy as np

class MarginLoss(object):
    pass


class Hinge(MarginLoss):
    @staticmethod
    def f(margin):
        margin = np.asarray(margin)
        return np.maximum(0, 1 - margin)

    @staticmethod
    def df(margin):
        margin = np.asarray(margin)
        return -1.0 * (margin < 1)


class L2Half(MarginLoss):
    @staticmethod
    def f(margin):
        margin = np.asarray(margin)
        return np.maximum(0, 1 - margin) ** 2

    @staticmethod
    def df(margin):
        margin = np.asarray(margin)
        return 2 * np.minimum(0, margin - 1)


class L2Huber(MarginLoss):
    @staticmethod
    def f(margin):
        margin = np.asarray(margin)
        if margin.ndim > 0:
            return np.piecewise(margin,
                    [margin < -1, (-1 <= margin) & (margin < 1), 1 <= margin],
                    [lambda x: -4. * x, lambda x: (1 - x) ** 2, 0.])
        else:
            if margin < -1: return -4. * margin
            if margin < 1: return (1 - margin) ** 2
            return 0.0

    @staticmethod
    def df(margin):
        return np.clip(2 * (margin - 1), -4, 0)


def loss_obj(obj):
    """
    Factory method turning `obj` into a MarginLoss either by pass-through or
    name lookup.
    """
    if isinstance(obj, basestring):
        _obj = globals()[obj]
    else:
        _obj = obj
    if not (isinstance(_obj, MarginLoss) or issubclass(_obj, MarginLoss)):
        raise TypeError('object does not name a loss function',
                obj)
    return _obj



