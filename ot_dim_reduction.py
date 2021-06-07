import ot.dr
import numpy as np
"""import tensorflow.compat.v1 as tf 

tf.disable_v2_behavior()
"""

def ot_dimension_reduction(X, y, reg=1):
    p = 50
    k = 10
    maxiter = 10

    # TODO tune p and reg !!!

    P0 = np.random.randn(X.shape[1], p)

    P0 /= np.sqrt(np.sum(P0 ** 2, 0, keepdims=True))

    Pwda, projwda = ot.dr.wda(X, y, p, reg, k, maxiter=maxiter, P0=P0)

    return Pwda, projwda


def dimension_reduction(X, projwda):
    return projwda(X)


def reverse_dimension_reduction(X, Pwda):
    mx = np.mean(X)
    return (X - mx.reshape((1, -1))).dot(Pwda.T)
