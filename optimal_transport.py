import ot
from reweighted_uot import WeightedUnbalancedSinkhornTransport
from utils import *


def ot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    """
    Function to compute the transport plan and transport targets into Source (or reverse)
    :param param_ot: hyperparameters of the transport
    :param X_source: Source features
    :param y_source: Source labels
    :param X_target: Target features
    :param transpose: boolean set by default to True : the X_target is transported in the Source domain
    if False : the sources are transported in the Target domain (classic transport)
    :return: Return the source features transported into the target if target_to_source = False
            Return the target features transported into the source if target_to_source = True
    """
    transport = ot.da.SinkhornLpl1Transport(reg_e=param_ot['reg_e'], reg_cl=param_ot['reg_cl'],
                                            norm="median")  # , log=True, verbose=True)
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)
    if transpose:
        transp_Xt = transport.inverse_transform(Xt=X_target)
        return transp_Xt
    else:
        transp_Xs = transport.transform(Xs=X_source, ys=y_source, Xt=X_target)
        return transp_Xs


def jmlot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    mapping = ot.da.MappingTransport(kernel="gaussian", mu=param_ot['mu'], sigma=param_ot['sigma'], eta=param_ot['eta'])
    mapping.fit(Xs=X_source, Xt=X_target)
    return mapping.transform(Xs=X_source)


def uot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    """
    Function to compute the transport plan and transport targets into Source (or reverse) with Unbalanced OT
    :param param_ot: hyperparameters of the transport
    :param X_source: Source features
    :param y_source: Source labels
    :param X_target: Target features
    :param transpose: boolean set by default to True : the X_target is transported in the Source domain
    if False : the sources are transported in the Target domain (classic transport)
    :return: Return the source features transported into the target if target_to_source = False
            Return the target features transported into the source if target_to_source = True
    """
    # https://pythonot.github.io/_modules/ot/da.html#SinkhornLpl1Transport
    # https://pythonot.github.io/gen_modules/ot.unbalanced.html
    # https://pythonot.github.io/_modules/ot/unbalanced.html#sinkhorn_knopp_unbalanced

    transport = ot.da.UnbalancedSinkhornTransport(reg_e=param_ot['reg_e'], reg_m=param_ot['reg_m'], verbose=True,
                                                  log=True)
    # default use sinkhorn_knopp_unbalanced
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)

    df = pd.DataFrame(transport.coupling_)
    df.to_csv("coupling_matrix_uot.csv")

    if transpose:
        # transport.coupling_ = transport.coupling_ / np.mean(transport.coupling_, axis=0)
        transp_Xt = transport.inverse_transform(Xt=X_target)
        return transp_Xt
    else:
        transp_Xs = transport.transform(Xs=X_source)
        return transp_Xs


def reweighted_uot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    transport = WeightedUnbalancedSinkhornTransport(reg_e=param_ot['reg_e'], reg_m=param_ot['reg_m'])
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)
    if transpose:
        transp_Xt = transport.inverse_transform(Xt=X_target)
        return transp_Xt
    else:
        transp_Xs = transport.transform(Xs=X_source)
        return transp_Xs


def jcpot_adaptation(X_source, y_source, X_target, param_ot, transpose=True):
    """
    OT for multi-source target shift,
    paper : "Optimal transport for multi-source domain adaptation under target shift" (Redko et al., 2019)
    :param param_ot: hyperparameters of the transport
    :param X_source: Source features
    :param y_source: Source labels
    :param X_target: Target features
    :param transpose: boolean set by default to True : the X_target is transported in the Source domain
    if False : the sources are transported in the Target domain (classic transport)
    :return: Return the source features transported into the target if target_to_source = False
            Return the target features transported into the source if target_to_source = True
    """
    transport = ot.da.JCPOTTransport(reg_e=param_ot['reg_e'], max_iter=1000, verbose=True, log=True)
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)
    transp_Xs = transport.transform(Xs=X_source, ys=y_source, Xt=X_target)
    return transp_Xs


def generateSubset2(X, Y, p):
    """
    This function should not be used on target true label because the proportion of classes are not available.
    :param X: Features
    :param Y: Labels
    :param p: Percentage of data kept.
    :return: Subset of X and Y with same proportion of classes.
    """
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:int(p * len(idxClass))])
    return X[idx], Y[idx]


def generateSubset4(X, Y, X_2, Y_2, p):
    """
    This function should not be used on target true label because the proportion of classes are not available.
    :param Y_2:
    :param X_2:
    :param X: Features
    :param Y: Labels
    :param p: Percentage of data kept.
    :return: Subset of X and Y with same proportion of classes.
    """
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:int(p * len(idxClass))])
    return X[idx], Y[idx], X_2[idx], Y_2[idx]


def ot_cross_validation(X_source, y_source, X_target, param_model_path, param_to_cross_valid, normalizer,
                        rescale, y_target=None, cv_with_true_labels=False,
                        transpose_plan=False, ot_type="OT", filename="",
                        duration_max=24, nb_training_iteration=8, gridsearch=True, cluster=1):
    """
    find the best hyperparameters for an optimal transport
    :param cv_with_true_labels:
    :param y_target:
    :param X_source:
    :param y_source:
    :param X_target:
    :param param_model: parameters of the model (eg. XGBoost)
    :param param_to_cross_valid: dictionary of parameters we want to cross valid
    :param transpose_plan: True to project targets in Source, False otherwise (classic OT)
    :param ot_type: values can be "UOT" or "OT"
    :param duration_max: maximum running time
    :param nb_training_iteration:
    :param gridsearch: if True a GridSearch is done to tune the parameters, otherwise it RandomSearch
    :return:
    """
    max_iteration = 1000
    possible_param_combination = []

    if gridsearch:
        possible_param_combination = create_grid_search_ot(param_to_cross_valid)
        max_iteration = len(possible_param_combination)

    param_train = dict([('reg_e', 0), ('reg_cl', 0)])
    time_start = time.time()
    nb_iteration = 0
    list_results = []
    list_results_cheat = []

    if cluster == 1:
        param_model = {'colsample_bytree': 0.4,
                       'eta': 0.1,
                       'gamma': 0.0,
                       'max_depth': 2,
                       'num_round': 1000,
                       'subsample': 0.8}
    if cluster == 2:
        param_model = {'colsample_bytree': 0.8,
                       'eta': 0.1,
                       'gamma': 2.0,
                       'max_depth': 1,
                       'num_round': 700,
                       'subsample': 0.6}
    else:
        param_model = import_hyperparameters("XGBoost", param_model_path)

    # param_model = cross_validation_model(X_source, y_source, hyperparameter_file=param_model_path, nbFoldValid=2)
    # param_model = import_hyperparameters("XGBoost", param_model_path)

    while time.time() - time_start < 3600 * duration_max and nb_iteration < max_iteration:
        np.random.seed(4896 * nb_iteration + 5272)
        if gridsearch and len(possible_param_combination) > 0:
            param_train = possible_param_combination[nb_iteration]
        else:  # Random search
            for key in param_to_cross_valid.keys():
                param_train[key] = param_to_cross_valid[key][np.random.randint(len(param_to_cross_valid[key]))]
        # try:
        for i in range(nb_training_iteration - 1):
            ic(param_train)
            # if we want to project the targets in the Source domain
            if transpose_plan:
                # Do the first adaptation (from source to target for the plan but adapt with the transpose)
                if ot_type == "OT":
                    trans_X_target = ot_adaptation(X_source, y_source, X_target, param_train, transpose=True)
                elif ot_type == "JCPOT":
                    trans_X_target = jcpot_adaptation(X_source, y_source, X_target, param_train, transpose=True)
                elif ot_type == "reweight_UOT":
                    trans_X_target = reweighted_uot_adaptation(X_source, y_source, X_target, param_train,
                                                               transpose=True)
                elif ot_type == "JMLOT":
                    trans_X_target = jmlot_adaptation(X_source, y_source, X_target, param_train,
                                                      transpose=True)
                else:  # Unbalanced OT
                    trans_X_target = uot_adaptation(X_source, y_source, X_target, param_train,
                                                    transpose=True)

                # Get pseudo labels
                if rescale:
                    rescaled_X_source = normalize(X_source, normalizer, True)
                    rescaled_trans_X_target = normalize(trans_X_target, normalizer, True)
                    # TODO
                    """rescaled_X_source = normalizer.inverse_transform(X_source)
                    rescaled_trans_X_target = normalizer.inverse_transform(trans_X_target)"""
                    trans_pseudo_y_target = predict_label(param_model, rescaled_X_source, y_source,
                                                          rescaled_trans_X_target)
                else:
                    trans_pseudo_y_target = predict_label(param_model, X_source, y_source, trans_X_target)
                if cv_with_true_labels and y_target is not None:
                    trans_pseudo_y_target_cheat = y_target
                else:
                    print("Warning inconsistence in parameters of ot_cross_validation : cv_with_true_labels is ",
                          "True but y_target is None")

                # Do the second adaptation (from target to source)
                # We don't use target_to_source = True, instead we reverse the target and source in parameters
                # bc we don't want to use the transpose of a plan here, just create a plan from Target to Source
                # trans2_X_target = ot_adaptation(trans_X_target, trans_pseudo_y_target, X_source, param_train)
                if ot_type == "OT":
                    trans2_X_target = ot_adaptation(X_target, trans_pseudo_y_target, X_source, param_train,
                                                    transpose=False)
                    if cv_with_true_labels and y_target is not None:
                        trans2_X_target_cheat = ot_adaptation(X_target, trans_pseudo_y_target_cheat, X_source,
                                                              param_train, transpose=False)
                elif ot_type == "JCPOT":
                    trans2_X_target = jcpot_adaptation(X_target, trans_pseudo_y_target, X_source, param_train,
                                                       transpose=False)
                    if cv_with_true_labels and y_target is not None:
                        trans2_X_target_cheat = jcpot_adaptation(X_target, trans_pseudo_y_target_cheat, X_source,
                                                                 param_train, transpose=False)
                elif ot_type == "reweight_UOT":
                    trans2_X_target = reweighted_uot_adaptation(X_target, trans_pseudo_y_target, X_source, param_train,
                                                                transpose=False)
                    if cv_with_true_labels and y_target is not None:
                        trans2_X_target_cheat = reweighted_uot_adaptation(X_target, trans_pseudo_y_target_cheat,
                                                                          X_source, param_train, transpose=False)
                elif ot_type == "JMLOT":
                    trans2_X_target = jmlot_adaptation(X_target, trans_pseudo_y_target, X_source, param_train,
                                                       transpose=False)
                    if cv_with_true_labels and y_target is not None:
                        trans2_X_target_cheat = jmlot_adaptation(X_target, trans_pseudo_y_target_cheat,
                                                                 X_source, param_train, transpose=False)
                else:  # Unbalanced OT
                    trans2_X_target = uot_adaptation(X_target, trans_pseudo_y_target, X_source, param_train,
                                                     transpose=False)
                    if cv_with_true_labels and y_target is not None:
                        trans2_X_target_cheat = uot_adaptation(X_target, trans_pseudo_y_target_cheat, X_source,
                                                               param_train, transpose=False)

                for j in range(nb_training_iteration):
                    if cv_with_true_labels and y_target is not None:
                        if ot_type == "JCPOT":
                            temp_trans2_X_target = np.append(trans2_X_target[0], trans2_X_target[1], axis=0)
                            temp_trans2_X_target = np.append(temp_trans2_X_target, trans2_X_target[2], axis=0)

                            temp_trans2_X_target_cheat = np.append(trans2_X_target_cheat[0],
                                                                   trans2_X_target_cheat[1], axis=0)
                            temp_trans2_X_target_cheat = np.append(temp_trans2_X_target_cheat,
                                                                   trans2_X_target_cheat[2], axis=0)
                            subset_trans2_X_target, subset_trans_pseudo_y_target, subset_trans2_X_target_cheat, \
                            subset_trans_pseudo_y_target_cheat = generateSubset4(temp_trans2_X_target,
                                                                                 trans_pseudo_y_target,
                                                                                 temp_trans2_X_target_cheat,
                                                                                 trans_pseudo_y_target_cheat, p=0.5)
                        else:
                            subset_trans2_X_target, subset_trans_pseudo_y_target, subset_trans2_X_target_cheat, \
                            subset_trans_pseudo_y_target_cheat = generateSubset4(trans2_X_target, trans_pseudo_y_target,
                                                                                 trans2_X_target_cheat,
                                                                                 trans_pseudo_y_target_cheat, p=0.5)
                    else:
                        if ot_type == "JCPOT":
                            temp_trans2_X_target = np.append(trans2_X_target[0], trans2_X_target[1], axis=0)
                            temp_trans2_X_target = np.append(temp_trans2_X_target, trans2_X_target[2], axis=0)
                            subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(temp_trans2_X_target,
                                                                                                   trans_pseudo_y_target,
                                                                                                   p=0.5)
                        else:
                            subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(trans2_X_target,
                                                                                                   trans_pseudo_y_target,
                                                                                                   p=0.5)

                    # = train_test_split(trans2_X_target, trans_pseudo_y_target, test_size=0.5, shuffle=True)
                    # = generateSubset2(trans2_X_target,trans_pseudo_y_target,p=0.5)

                    if rescale:
                        rescaled_X_source = normalize(X_source, normalizer, True)
                        rescaled_subset_trans2_X_target = normalize(subset_trans2_X_target, normalizer, True)
                        # TODO
                        """rescaled_X_source = normalizer.inverse_transform(X_source)
                        rescaled_subset_trans2_X_target = normalizer.inverse_transform(subset_trans2_X_target)"""
                        y_source_pred = predict_label(param_model,
                                                      rescaled_subset_trans2_X_target,
                                                      subset_trans_pseudo_y_target,
                                                      rescaled_X_source)

                        if cv_with_true_labels and y_target is not None:
                            rescaled_X_source_cheat = normalize(X_source, normalizer, True)
                            rescaled_subset_trans2_X_target_cheat = normalize(subset_trans2_X_target_cheat,
                                                                              normalizer, True)
                            y_source_pred_cheat = predict_label(param_model,
                                                                rescaled_subset_trans2_X_target_cheat,
                                                                subset_trans_pseudo_y_target_cheat,
                                                                rescaled_X_source_cheat)
                    else:
                        y_source_pred = predict_label(param_model,
                                                      subset_trans2_X_target,
                                                      subset_trans_pseudo_y_target,
                                                      X_source)
                        if cv_with_true_labels and y_target is not None:
                            y_source_pred_cheat = predict_label(param_model,
                                                                subset_trans2_X_target_cheat,
                                                                subset_trans_pseudo_y_target_cheat,
                                                                X_source)

                    precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
                    average_precision = 100 * average_precision_score(y_source, y_source_pred)

                    precision_cheat = 100 * float(sum(y_source_pred_cheat == y_source)) / len(y_source_pred_cheat)
                    average_precision_cheat = 100 * average_precision_score(y_source, y_source_pred_cheat)

                # add results + param for this loop to the pickle
                to_save = dict(param_train)
                to_save['precision'] = precision
                to_save['average_precision'] = average_precision
                list_results.append(to_save)

                to_save = dict(param_train)
                to_save['precision'] = precision_cheat
                to_save['average_precision'] = average_precision_cheat
                list_results_cheat.append(to_save)
            # if we want to project the sources in the Target domain (classic method)
            else:
                # First adaptation
                if ot_type == "OT":
                    trans_X_source = ot_adaptation(X_source, y_source, X_target, param_train,
                                                   transpose=False)
                elif ot_type == "reweight_UOT":
                    trans_X_source = reweighted_uot_adaptation(X_source, y_source, X_target, param_train,
                                                               transpose=False)
                elif ot_type == "JMLOT":
                    trans_X_source = jmlot_adaptation(X_source, y_source, X_target, param_train,
                                                      transpose=False)
                else:
                    trans_X_source = uot_adaptation(X_source, y_source, X_target, param_train,
                                                    transpose=False)

                # Get pseudo labels
                trans_pseudo_y_target = predict_label(param_model, trans_X_source, y_source, X_target)
                if cv_with_true_labels and y_target is not None:
                    precision_cheat = 100 * float(sum(trans_pseudo_y_target == y_target)) / len(trans_pseudo_y_target)
                    average_precision_cheat = 100 * average_precision_score(y_target, trans_pseudo_y_target)
                    to_save_cheat = dict(param_train)
                    to_save_cheat['precision'] = precision_cheat
                    to_save_cheat['average_precision'] = average_precision_cheat
                    list_results_cheat.append(to_save_cheat)
                else:
                    print("Warning inconsistence in parameters of ot_cross_validation : cv_with_true_labels is ",
                          "True but y_target is None")

                # Second adaptation
                if ot_type == "OT":
                    trans2_X_target = ot_adaptation(X_source=X_target, y_source=trans_pseudo_y_target,
                                                    X_target=X_source, param_ot=param_train,
                                                    transpose=False)
                elif ot_type == "reweight_UOT":
                    trans2_X_target = reweighted_uot_adaptation(X_source=X_target, y_source=trans_pseudo_y_target,
                                                                X_target=X_source, param_ot=param_train,
                                                                transpose=False)
                elif ot_type == "JMLOT":
                    trans2_X_target = jmlot_adaptation(X_source=X_target, y_source=trans_pseudo_y_target,
                                                       X_target=X_source, param_ot=param_train,
                                                       transpose=False)
                else:  # Unbalanced OT
                    trans2_X_target = uot_adaptation(X_source=X_target, y_source=trans_pseudo_y_target,
                                                     X_target=X_source, param_ot=param_train,
                                                     transpose=False)

                for j in range(nb_training_iteration):
                    subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(trans2_X_target,
                                                                                           trans_pseudo_y_target,
                                                                                           p=0.5)

                    # TODO remove this line if useless (slow the computation)
                    param_model = cross_validation_model(subset_trans2_X_target, subset_trans_pseudo_y_target,
                                                         hyperparameter_file=param_model_path, nbFoldValid=2)
                    # ic("CV model in CV OT:", param_model)

                    y_source_pred = predict_label(param_model,
                                                  subset_trans2_X_target,
                                                  subset_trans_pseudo_y_target,
                                                  X_source)
                    precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
                    average_precision = 100 * average_precision_score(y_source, y_source_pred)

                # add results + param for this loop to the pickle
                to_save = dict(param_train)
                to_save['precision'] = precision
                to_save['average_precision'] = average_precision
                list_results.append(to_save)

        time.sleep(1.)  # Allow us to stop the program with ctrl-C
        nb_iteration += 1
        if to_save:
            ic(nb_iteration, to_save)
        else:
            ic(nb_iteration)
    if cv_with_true_labels and y_target is not None:
        optimal_param = max(list_results, key=lambda val: val['average_precision'])
        optimal_param_cheat = max(list_results_cheat, key=lambda val: val['average_precision'])
        results = {'cv_res': list_results, 'cheat': list_results_cheat}
        f = gzip.open(filename, "wb")
        pickle.dump(results, f)
        f.close()
        return optimal_param, optimal_param_cheat
    else:
        f = gzip.open(filename, "wb")
        pickle.dump(list_results, f)
        f.close()
        optimal_param = max(list_results, key=lambda val: val['average_precision'])
        return optimal_param, None


def ot_cross_validation_jcpot(X_source, y_source, X_target, param_model, param_model_path, param_to_cross_valid,
                              filename="cv_jcpot", y_target=None, cv_with_true_labels=False,
                              duration_max=24, nb_training_iteration=8, gridsearch=True, cluster=1):
    max_iteration = 1000
    possible_param_combination = []

    if gridsearch:
        possible_param_combination = create_grid_search_ot(param_to_cross_valid)
        max_iteration = len(possible_param_combination)
        ic(len(possible_param_combination))

    param_train = dict([('reg_e', 0), ('reg_cl', 0)])
    time_start = time.time()
    nb_iteration = 0
    list_results = []
    list_results_cheat = []

    concat_X_source = np.append(X_source[0], X_source[1], axis=0)
    concat_X_source = np.append(concat_X_source, X_source[2], axis=0)

    if cluster == 1:
        param_model = {'colsample_bytree': 0.4,
                       'eta': 0.1,
                       'gamma': 0.0,
                       'max_depth': 2,
                       'num_round': 1000,
                       'subsample': 0.8}
    elif cluster == 2:
        param_model = {'colsample_bytree': 0.8,
                       'eta': 0.1,
                       'gamma': 2.0,
                       'max_depth': 1,
                       'num_round': 700,
                       'subsample': 0.6}
    else:
        param_model = import_hyperparameters("XGBoost", param_model_path)

    # param_model = cross_validation_model(concat_X_source, y_source, hyperparameter_file=param_model, nbFoldValid=2)
    # param_model = import_hyperparameters("XGBoost", param_model)
    bst = None
    if cv_with_true_labels:
        bst = get_xgboost_model(param_model, concat_X_source, y_source)
    ic(param_model)
    while time.time() - time_start < 3600 * duration_max and nb_iteration < max_iteration:
        np.random.seed(4896 * nb_iteration + 5272)
        if gridsearch and len(possible_param_combination) > 0:
            param_train = possible_param_combination[nb_iteration]
        else:  # Random search
            for key in param_to_cross_valid.keys():
                param_train[key] = param_to_cross_valid[key][np.random.randint(len(param_to_cross_valid[key]))]
        for i in range(nb_training_iteration):
            ic(param_train)
            # First adaptation
            trans_X_source = jcpot_adaptation(X_source, y_source, X_target, param_train,
                                              transpose=False)
            # Get pseudo labels
            concat_trans_X_source = np.append(trans_X_source[0], trans_X_source[1], axis=0)
            concat_trans_X_source = np.append(concat_trans_X_source, trans_X_source[2], axis=0)
            trans_pseudo_y_target = predict_label(param_model, concat_trans_X_source, y_source, X_target)

            if cv_with_true_labels and y_target is not None:
                """concat_X_source = np.append(X_source[0], X_source[1], axis=0)
                concat_X_source = np.append(concat_X_source, X_source[2], axis=0)
                y_source_pred_cheat = predict_label_with_xgboost(bst, concat_X_source, y_source, X_target)"""

                precision_cheat = 100 * float(sum(trans_pseudo_y_target == y_target)) / len(trans_pseudo_y_target)
                average_precision_cheat = 100 * average_precision_score(y_target, trans_pseudo_y_target)
                to_save_cheat = dict(param_train)
                to_save_cheat['precision'] = precision_cheat
                to_save_cheat['average_precision'] = average_precision_cheat
                list_results_cheat.append(to_save_cheat)
            else:
                print("Warning inconsistence in parameters of ot_cross_validation : cv_with_true_labels is ",
                      "True but y_target is None")

            # Second adaptation

            # TODO check the direction of the transport, we need to get trans2_X_target[0] I think
            trans2_X_target = jcpot_adaptation(X_source=[X_target], y_source=trans_pseudo_y_target,
                                               X_target=concat_X_source, param_ot=param_train,
                                               transpose=False)[0]

            for j in range(nb_training_iteration):
                subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(trans2_X_target,
                                                                                       trans_pseudo_y_target,
                                                                                       p=0.5)

                # TODO remove this line if useless (slow the computation)
                # param_model = cross_validation_model(subset_trans2_X_target, subset_trans_pseudo_y_target, hyperparameter_file=param_model_path, nbFoldValid=2)
                # ic("CV model in CV OT:", param_model)

                y_source_pred = predict_label(param_model,
                                              subset_trans2_X_target,
                                              subset_trans_pseudo_y_target,
                                              concat_X_source)
                precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
                average_precision = 100 * average_precision_score(y_source, y_source_pred)

            # add results + param for this loop to the pickle
            to_save = dict(param_train)
            to_save['precision'] = precision
            to_save['average_precision'] = average_precision
            list_results.append(to_save)
        time.sleep(1.)  # Allow us to stop the program with ctrl-C
        nb_iteration += 1
        if to_save:
            ic(nb_iteration, to_save)
        else:
            ic(nb_iteration)
    if cv_with_true_labels and y_target is not None:
        optimal_param = max(list_results, key=lambda val: val['average_precision'])
        optimal_param_cheat = max(list_results_cheat, key=lambda val: val['average_precision'])
        results = {'cv_res': list_results, 'cheat': list_results_cheat}
        f = gzip.open(filename, "wb")
        pickle.dump(results, f)
        f.close()
        return optimal_param, optimal_param_cheat
    else:
        f = gzip.open(filename, "wb")
        pickle.dump(list_results, f)
        f.close()
        optimal_param = max(list_results, key=lambda val: val['average_precision'])
        return optimal_param, None
