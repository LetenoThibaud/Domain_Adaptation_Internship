# Implementation based on POT: Python Optimal Transport.

import warnings
import numpy as np
from ot.da import distribution_estimation_uniform, BaseTransport
from ot.utils import check_params


class WeightedUnbalancedSinkhornTransport(BaseTransport):
    def __init__(self, reg_e=1., reg_m=0.1, method='sinkhorn',
                 max_iter=10, tol=1e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10):
        self.reg_e = reg_e
        self.reg_m = reg_m
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys, Xt=None, yt=None):
        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            super(WeightedUnbalancedSinkhornTransport, self).fit(Xs, ys, Xt, yt)

            returned_ = weighted_sinkhorn_unbalanced(
                a=self.mu_s, b=self.mu_t, a_label=ys, M=self.cost_,
                reg=self.reg_e, reg_m=self.reg_m, method=self.method,
                numItermax=self.max_iter, stopThr=self.tol,
                verbose=self.verbose, log=self.log)

            # deal with the value of log
            if self.log:
                self.coupling_, self.log_ = returned_
            else:
                self.coupling_ = returned_
                self.log_ = dict()

        return self


def weighted_sinkhorn_unbalanced(a, b, a_label, M, reg, reg_m, method='sinkhorn', numItermax=1000,
                                 stopThr=1e-6, verbose=False, log=False, **kwargs):
    if method.lower() == 'sinkhorn':
        return weighted_sinkhorn_knopp_unbalanced(a, b, a_label, M, reg, reg_m,
                                                  numItermax=numItermax,
                                                  stopThr=stopThr, verbose=verbose,
                                                  log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def weighted_sinkhorn_knopp_unbalanced(a, b, a_label, M, reg, reg_m, numItermax=1000,
                                       stopThr=1e-6, verbose=False, log=False, **kwargs):
    # a the distribution of the source
    # b the distribution of the target
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, 1)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    # fi = reg_m / (reg_m + reg)

    fi_0 = reg_m["0"] / (reg_m["0"] + reg)
    fi_1 = reg_m["1"] / (reg_m["1"] + reg)
    reg_m_mean = (reg_m["0"] + reg_m["1"])/2
    fi_mean = reg_m_mean / (reg_m_mean + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = np.where(a_label == 1, (a/Kv)**fi_1, (a/Kv)**fi_0)
        # u = (a / Kv) ** fi

        Ktu = K.T.dot(u)
        v = (b / Ktu) ** fi_mean
        # v = (b / Ktu) ** fi

        if (np.any(Ktu == 0.)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
        err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)

    if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]
