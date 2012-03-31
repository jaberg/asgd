import numpy as np
import asgd

def test_example():
    # -- True model (w, b)
    w = np.random.randn(5)       # weights
    b = np.random.randn()        # bias

    # -- fake dataset (X, y)
    X = np.random.randn(100, 5)  # 100 fake 5-d examples
    y = np.sign(np.dot(X, w) + b)

    svm = asgd.LinearSVM(l2_regularization=0.01, verbose=True)
    svm.fit(X, y)
    print w, svm.svm.weights
    print b, svm.svm.bias
    print np.mean(svm.predict(X) == y)
