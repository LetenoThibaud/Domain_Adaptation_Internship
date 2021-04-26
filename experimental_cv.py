import warnings

import numpy as np
from ot.da import distribution_estimation_uniform, BaseTransport
from ot.utils import check_params
import icecream as ic
from main import *


def reweighted_uot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    transport = UnbalancedSinkhornTransport(reg_e=param_ot['reg_e'], reg_m=param_ot['reg_m'])
    # default use sinkhorn_knopp_unbalanced
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)
    if transpose:
        transp_Xt = transport.inverse_transform(Xt=X_target)
        return transp_Xt
    else:
        transp_Xs = transport.transform(Xs=X_source)
        return transp_Xs


class UnbalancedSinkhornTransport(BaseTransport):
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

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            super(UnbalancedSinkhornTransport, self).fit(Xs, ys, Xt, yt)

            returned_ = sinkhorn_unbalanced(
                a=self.mu_s, b=self.mu_t, M=self.cost_,
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


def sinkhorn_unbalanced(a, b, M, reg, reg_m, method='sinkhorn', numItermax=1000,
                        stopThr=1e-6, verbose=False, log=False, **kwargs):
    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m,
                                         numItermax=numItermax,
                                         stopThr=stopThr, verbose=verbose,
                                         log=log, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, **kwargs):
    # a the distribution of the source
    # b the distribution of the target
    ic(a)
    a = np.asarray(a, dtype=np.float64)
    ic(a)
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
        ic(a)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    ic(a)
    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        u = (a / Kv) ** fi
        Ktu = K.T.dot(u)
        v = (b / Ktu) ** fi

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


def expe_main():
    seed = 1
    algo = "XGBoost"
    results = {}
    filename = "results/experimental_reweighted_uot.pklz"
    for dataset in ['abalone20']:  # , 'abalone17', 'satimage', 'abalone8']:  # ['abalone8']:  #

        start = time.time()
        X, y = data_recovery(dataset)
        dataset_name = dataset
        pctPos = 100 * len(y[y == 1]) / len(y)
        dataset = "{:05.2f}%".format(pctPos) + " " + dataset
        results["expe"] = {}
        print(dataset)
        np.random.seed(seed)
        random.seed(seed)

        # import the tuned parameters of the model for this dataset
        params_model = import_hyperparameters(dataset_name, "hyperparameters_toy_dataset.csv")
        param_transport = dict()

        # Split the dataset between the source and the target(s)
        # TODO for UOT implementation :
        #  create unbalance during the split between source and target !
        Xsource, Xtarget, ysource, ytarget = train_test_split(X, y, shuffle=True,
                                                              stratify=y,
                                                              random_state=1234,
                                                              test_size=0.51)
        # Keep a clean backup of Xtarget before degradation.
        Xclean = Xtarget.copy()
        # for loop -> degradation of the target
        # 3 features are deteriorated : the 2nd, the 3rd and the 4th
        for feat, coef in [(2, 0.1), (3, 10), (4, 0)]:
            # for features 2 and 3, their values are multiplied by a coefficient
            # resp. 0.1 and 10
            if coef != 0:
                Xtarget[:, feat] = Xtarget[:, feat] * coef
            # for feature 4, some of its values are (randomly) set to 0
            else:
                Xtarget[np.random.choice(len(Xtarget), int(len(Xtarget) / 2)), feat] = 0

        param_transport = {'reg_e': 0.5, 'reg_m': 0.1}
        Xtarget = reweighted_uot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose=True)


        Xtrain, Xtest, ytrain, ytest = train_test_split(Xsource, ysource,
                                                        shuffle=True,
                                                        random_state=3456,
                                                        stratify=ysource,
                                                        test_size=0.3)

        apTrain, apTest, apClean, apTarget = applyAlgo(algo, params_model,
                                                       Xtrain, ytrain,
                                                       Xtest, ytest,
                                                       Xtarget, ytarget,
                                                       Xclean)

        results["expe"][algo] = (apTrain, apTest, apClean, apTarget, params_model, param_transport,
                                 time.time() - start)
        print(dataset, "expe", "Train AP {:5.2f}".format(apTrain),
              "Test AP {:5.2f}".format(apTest),
              "Clean AP {:5.2f}".format(apClean),
              "Target AP {:5.2f}".format(apTarget), params_model, param_transport,
              "in {:6.2f}s".format(time.time() - start))

        if not os.path.exists("results"):
            try:
                os.makedirs("results")
            except:
                pass
        if filename == "":
            filename = f"./results/res{seed}.pklz"
        f = gzip.open(filename, "wb")
        pickle.dump(results, f)
        f.close()
