import numpy as np
import matplotlib.pyplot as plt

from asgd.lossfns import Hinge, L2Half, L2Huber

def assert_loss_match(data, obj, show=True):
    if 1 or show:
        x = np.linspace(-3, 3, 100)
        print 'X', x
        plt.plot(x, obj.f(x))
        plt.plot(x, obj.df(x))
        plt.show()

    y = obj.f(data[:, 0])
    dy = obj.df(data[:, 0])

    assert np.allclose(y, data[:, 1]), (data, y)
    assert np.allclose(dy, data[:, 2]), (data, dy)


def test_hinge():
    margin_y_dy = np.asarray([
            [-10, 11, -1],
            [-2.5, 3.5, -1],
            [-1.5, 2.5, -1],
            [-.5, 1.5, -1],
            [0, 1, -1],
            [.5, .5, -1],
            [.99, .01, -1],
            [1.0, 0, 0],
            [2.0, 0, 0],
            ])
    assert_loss_match(margin_y_dy, Hinge, show=False)


def test_l2half():
    margin_y_dy = np.asarray([
        [-10,.5 * 11 ** 2, -11],
        [-2.5,.5 * 3.5 ** 2, -3.5],
        [-1.5,.5 * 2.5 ** 2, -2.5],
        [-.5,.5 * 1.5 ** 2, -1.5],
        [0,.5, -1],
        [.5, .125, -.5],
        [1.0, 0, 0],
        [2.0, 0, 0],
        ])

    margin_y_dy[:, 1:] *= 2
    assert_loss_match(margin_y_dy, L2Half, show=False)


def test_l2huber():
    margin_y_dy = np.asarray([
        [-10, 20, -2],
        [-2.5, 5, -2],
        [-1.5, 3, -2],
        [-.5,.5 * 1.5 ** 2, -1.5],
        [0,.5, -1],
        [.5, .125, -.5],
        [1.0, 0, 0],
        [2.0, 0, 0],
        ])

    margin_y_dy[:, 1:] *= 2
    assert_loss_match(margin_y_dy, L2Huber, show=False)
