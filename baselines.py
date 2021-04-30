from optimal_transport import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy.linalg import sqrtm

# -------------Subspace Alignment-------------#


def sa_adaptation(X_source, X_target, param_transport, transpose=True):
    if transpose:  # we want to project targets in Source
        # Generation of the subset
        pcaS = PCA(n_components=param_transport["d"]).fit(X_source)
        pcaT = PCA(n_components=param_transport["d"]).fit(X_target)
        # XS the source subspace
        XS = np.transpose(pcaS.components_)
        # XT the target subspace
        XT = np.transpose(pcaT.components_)
        # Xa the target aligned source system coordinates Xa = XT.XT'.XS
        # since M = XT'.XS (M the transformation matrix from XT to XS)
        Xa = XT.dot(np.transpose(XT)).dot(XS)
        sourceAdapted = X_source.dot(XS)
        targetAdapted = X_target.dot(Xa)  # we align target basis vectors with the source ones
    else:
        # Generation of the subset
        pcaS = PCA(n_components=param_transport["d"]).fit(X_source)
        pcaT = PCA(n_components=param_transport["d"]).fit(X_target)
        # XS the source subspace
        XS = np.transpose(pcaS.components_)
        # XT the target subspace
        XT = np.transpose(pcaT.components_)
        # Xa the target aligned source system coordinates Xa = XS.XS'.XT
        # since M = XS'.XT (M the transformation matrix from XS to XT)
        Xa = XS.dot(np.transpose(XS)).dot(XT)
        sourceAdapted = X_source.dot(Xa)  # we align source basis vectors with the target ones
        targetAdapted = X_target.dot(XT)

    return sourceAdapted, targetAdapted


def components_analysis_based_method_cross_validation(X_source, y_source, X_target, param_model, transport_type="SA",
                                                      algo='XGBoost'):
    """
    Tune the subspace dimensionality parameter d.
    :param transport_type:
    :param algo:
    :param X_source:
    :param y_source:
    :param X_target:
    :param param_model:
    :param transpose:
    :param nb_training_iteration:
    :return:
    """
    # Seach for the the max value of d we want to consider as best parameter
    # Compute all the principal components for the two domains
    pcaS = PCA().fit(X_source)
    pcaT = PCA().fit(X_target)
    deviation = []
    upper_bound = []
    delta = 0.1
    gamma = 1000

    for d in range(pcaS.n_components_ - 1):
        # compute the deviation lambda^min_d - lambda^min_d+1 for all possibles d values
        dev = min(pcaS.singular_values_[d] - pcaS.singular_values_[d + 1],
                  pcaT.singular_values_[d] - pcaT.singular_values_[d + 1])
        # compute the upper bound for all possibles d
        bound = (1 + np.math.sqrt(np.math.log(2 / delta, 10) / 2)) * (
                16 * np.math.pow(d, 3 / 2) / gamma * np.math.sqrt(pcaS.n_features_))
        deviation.append(dev)
        upper_bound.append(bound)

    fig = plt.figure()
    X = np.arange(pcaS.n_features_ - 1)
    # print(X, deviation, upper_bound)
    plt.plot(X, deviation, 'orange', label='difference in consecutive eigenvalues')
    plt.plot(X, upper_bound, 'g', label='upper bound')
    plt.legend()
    # plt.show()

    # find d_max such that for all d > d_max : upper_bound > deviation
    d_max = 1
    for i in range(pcaS.n_features_ - 1):
        if deviation[i] >= upper_bound[i]:
            d_max = i + 1

    # Test the quality of d for all d in [1; d_max] by testing it on the source
    # paper : consider the subspaces of dim d=1:dmax and select d* that minimizes the classification error
    #        using a 2-fold CV (on X_source)
    nbFoldValid = 10
    param_transport = dict()
    validParam = dict()
    for d in range(1, d_max):
        skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        param_transport["d"] = d
        if transport_type == "TCA":
            sourceAdapted, targetAdapted = tca_adaptation(X_source, X_target, param_transport)
        else:  # transport_type == "SA"
            sourceAdapted, targetAdapted = sa_adaptation(X_source, X_target, param_transport, transpose=False)
        ap_score = []
        for i in range(10):
            # fold_train, fold_valid = foldsTrainValid[iFoldVal]
            # 2-fold on source examples
            train_X, valid_X, train_y, valid_y = train_test_split(sourceAdapted, y_source, train_size=0.5, shuffle=True)
            predicted_y_valid = predict_label(param_model, train_X, train_y, valid_X, algo)
            average_precision = 100 * average_precision_score(valid_y, predicted_y_valid)
            ap_score.append(average_precision)
        validParam[d] = np.mean(ap_score)

    idx_max_value = int(np.argmax(list(validParam.values())))
    d_optimal = list(validParam.keys())[idx_max_value]
    return d_optimal


# -------------CORAL (CORelation ALignment)-------------#
def coral_adaptation(X_source, X_target, transpose=True):
    """
    :param X_source:
    :param X_target:
    :param transpose:
    :return:
    """
    # Classic CORAL adaptation (align the source distribution to the target one)
    if not transpose:
        Cs = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1])
        Ct = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1])
        Ds = X_source.dot(np.linalg.inv(np.real(sqrtm(Cs))))  # whitening source
        Ds = Ds.dot(np.real(sqrtm(Ct)))  # re-coloring with target covariance
        adapted_source = Ds
        adapted_target = X_target  # target remained unchanged
        return adapted_source, adapted_target
    # CORAL adaptation from Target to Source (align the target distribution to the source one)
    else:
        Cs = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1])
        Ct = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1])
        Dt = X_target.dot(np.linalg.inv(np.real(sqrtm(Ct))))  # whitening target
        Dt = Dt.dot(np.real(sqrtm(Cs)))  # re-coloring with source covariance
        adapted_source = X_source  # source remained unchanged
        adapted_target = Dt
        return adapted_source, adapted_target


# -------------TCA (Transport Components Analysis)------------- #
def tca_adaptation(X_source, X_target, param):
    # Note that there is no difference between the classic transport
    # and the solution where we aim at transporting the targets
    # into the source domain for this baseline, both the targets and
    # the sources are transported in an extra subspace
    d = param["d"]  # subspace dimension
    Ns = X_source.shape[0]
    Nt = X_target.shape[0]
    # L = [Lij] >= 0
    L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
    L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
    L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
    L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
    L_up = np.hstack((L_ss, L_st))
    L_down = np.hstack((L_ts, L_tt))
    L = np.vstack((L_up, L_down))
    # K the kernel map
    X = np.vstack((X_source, X_target))
    K = np.dot(X, X.T)  # linear kernel
    # H the centering matrix
    H = (np.identity(Ns + Nt) - 1. / (Ns + Nt) * np.ones((Ns + Nt, 1)) *
         np.ones((Ns + Nt, 1)).T)
    inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
    D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
    W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
    sourceAdapted = np.dot(K[:Ns, :], W)  # project source
    targetAdapted = np.dot(K[Ns:, :], W)  # project target
    return sourceAdapted, targetAdapted
