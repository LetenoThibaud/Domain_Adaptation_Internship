from optimal_transport import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from main import applyAlgo


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


def sa_cross_validation(X_source, y_source, X_target, param_model,
                        transpose=True):
    """
    We tune the subspace dimensionality d.
    :param X_source:
    :param y_source:
    :param X_target:
    :param param_model:
    :param transpose:
    :param nb_training_iteration:
    :return:
    """

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
    plt.show()

    # find d_max such that for all d > d_max : upper_bound > deviation
    d_max = 1
    for i in range(pcaS.n_features_ - 1):
        if deviation[i] >= upper_bound[i]:
            d_max = i + 1
    # paper : consider the subspaces of dim d=1:dmax and select d* that minimizes the classification error
    #        using a 2-fold CV (on X_source)
    nbFoldValid = 10
    param_transport = dict()
    validParam = dict()
    for d in range(d_max):
        skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        param_transport["d"] = d
        sourceAdapted, targetAdapted = sa_adaptation(X_source, X_target, param_transport, transpose)
        foldsTrainValid = list(skf.split(sourceAdapted, y_source))
        ap_score = []
        for iFoldVal in range(nbFoldValid):
            fold_train, fold_valid = foldsTrainValid[iFoldVal]
            ic(fold_train, fold_valid, sourceAdapted[fold_train], sourceAdapted[fold_valid])
            ic(param_model)
            predicted_y_valid = predict_label(param_model, sourceAdapted[fold_train], y_source[fold_train],
                                              sourceAdapted[fold_valid], algo='XGBoost')

            average_precision = 100 * average_precision_score(y_source[fold_valid], predicted_y_valid)
            ap_score.append(average_precision)
        validParam[d] = np.mean(ap_score)
    d_optimal = list(validParam.keys())[list(validParam.values()).index(np.argmax(validParam.values()))]
    # TODO export to csv !!
    return d_optimal

# -------------  -------------#
