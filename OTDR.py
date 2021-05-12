import ot.dr
import numpy as np


def ot_dimension_reduction(Xsource, Xtarget, ysource, reg_reduction="0.05", p=2, method="FDA"):
    p = 2
    reg = 1e0
    k = 10
    maxiter = 100

    P0 = np.random.randn(Xsource.shape[1], p)

    P0 /= np.sqrt(np.sum(P0 ** 2, 0, keepdims=True))

    Pwda, projwda = ot.dr.wda(Xsource, ysource, p, reg, k, maxiter=maxiter, P0=P0)

    return projwda(Xsource), projwda(Xtarget), Pwda


def reverse_dimension_reduction(X, Popt):
    mx = np.mean(X)
    return (X - mx.reshape((1, -1))).dot(Popt.T)

