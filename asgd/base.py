import numpy as np

class BinaryClassifier(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.n_features = len(self.weights)
        self.n_classes = 2

    def decisions(self, X):
        X = np.asarray(X)
        return np.dot(self.weights, X.T) + self.bias

    def predict(self, X):
        return np.sign(self.decisions(X))

    def __call__(self, X):
        return self.predict(X)


class MultiClassifier(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.n_features, self.n_classes = weights.shape
        if len(bias) != self.n_classes:
            raise ValueError('bias has wrong shape for %i classes' %
                    self.n_classes, bias.shape)

    def decisions(self, X):
        X = np.asarray(X)
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self.decisions(X).argmax(axis=1)

    def __call__(self, X):
        return self.predict(X)


def classifier(n_classes, n_features, dtype='float64'):
    """Return an un-trained, null classifier.
    """
    if n_classes < 2:
        raise ValueError('need at least 2 classes')
    elif n_classes == 2:
        weights = np.zeros((n_features,), dtype=dtype)
        bias = np.asarray(0, dtype=dtype)
        return BinaryClassifier(weights, bias)
    else:
        weights = np.zeros((n_features, n_classes), dtype=dtype)
        bias = np.zeros(n_classes, dtype=dtype)
        return MultiClassifier(weights, bias)


def classifier_from_weights(weights, bias):
    if weights.ndim == 2:
        return MultiClassifier(weights, bias)
    else:
        return BinaryClassifier(weights, bias)

