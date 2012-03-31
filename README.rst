ASGD (jaberg fork)
==================

This package provides several primal linear SVM solvers, which are appropriate
for dense design matrices (feature matrices).  The main entry point for new users is
the `asgd.LinearSVM` object which is a sklearn-compatible class that provides
a somewhat uniform interface to various internal and external solvers.


.. code:: python

    import numpy as np

    # -- True model (w, b)
    w = np.random.randn(5)       # weights
    b = np.random.randn()        # bias

    # -- fake dataset (X, y)
    X = np.random.randn(100, 5)  # 100 fake 5-d examples
    y = np.sign(np.dot(X, w) + b)

    svm = asgd.LinearSVM(l2_regularization=0.01)
    svm.fit(X, y)
    print w, svm.svm.weights
    print b, svm.svm.bias
    print np.mean(svm.predict(X) == y)




References:
===========

*   `"Towards Optimal One Pass Large Scale Learning with Averaged Stochastic
    Gradient Descent"
    <http://arxiv.org/abs/1107.2490>`_
    Wei Xu (2011)

*   `"Large-scale Image Classification: Fast Feature Extraction and SVM Training"
    <http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf>`_
    Yuanqing Lin, Fengjun Lv, Shenghuo Zhu, Ming Yang, Timothee Cour, Kai Yu,
    Liangliang Cao, and Thomas Huang (CVPR 2011)

*   `"Large-Scale Machine Learning with Stochastic Gradient Descent"
    <http://leon.bottou.org/publications/pdf/compstat-2010.pdf>`_
    Leon Bottou (2010)

*   `"Non-Asymptotic Analysis of Stochastic Approximation Algorithms for
    Machine Learning"
    <http://hal.archives-ouvertes.fr/docs/00/60/80/41/PDF/gradsto_hal.pdf>`_
    Francis Bach, Eric Moulines (NIPS/HAL 2011)
