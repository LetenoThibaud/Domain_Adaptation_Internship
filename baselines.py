from optimal_transport import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
                        transpose=True, nb_training_iteration=10):
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

    # TODO : pb find how to keep d_max such that d_max is the value after the one upper_bound > deviation
    # TODO 2 : consider the subspaces of dim d=1:dmax and select d* that minimizes the classification error
    #  using a 2-fold CV (on X_source)


    # # XS the source subspace
    # XS = np.transpose(pcaS.components_)
    # # XT the target subspace
    # XT = np.transpose(pcaT.components_)
    # # Xa the target aligned source system coordinates Xa = XS.XS'.XT
    # # since M = XS'.XT (M the transformation matrix from XS to XT)
    # Xa = XS.dot(np.transpose(XS)).dot(XT)
    # sourceAdapted = X_source.dot(Xa)  # we align source basis vectors with the target ones
    # targetAdapted = X_target.dot(XT)




    """param_transport = dict()
    nb_features = 3  # TODO X_source
    # we want to tune d the dimension of the
    for d in range(1, nb_features, 1):
        param_transport['d'] = d
        sourceAdapted, targetAdapted = sa_adaptation(X_source, X_target, param_transport, transpose)
        pred_targets_labels = predict_label(param_model, sourceAdapted, y_source, targetAdapted)

        for j in range(10):
            subset_adapted_target, subset_pred_y_target = generateSubset2(targetAdapted,
                                                                          pred_targets_labels,
                                                                          p=0.5)
            # ic(subset_trans2_X_target)
            # ic(subset_trans_pseudo_y_target)
            y_source_pred = predict_label(param_model,
                                          subset_adapted_target,
                                          subset_pred_y_target,
                                          X_source)
            precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
            average_precision = 100 * average_precision_score(y_source, y_source_pred)

        # add results + param for this loop to the pickle
        to_save = dict(param_train)
        to_save['precision'] = precision
        to_save['average_precision'] = average_precision
        list_results.append(to_save)"""
