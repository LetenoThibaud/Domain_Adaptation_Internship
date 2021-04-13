from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from optimal_transport import *


def double_cross_valid(X_source, y_source, X_target, param_model, param_ot, pickle_name,
                       duration_max=24, nb_training_iteration=10, search_method="GridSearch"):
    max_iteration = 1000
    possible_param_combination = []
    if search_method == "GridSearch":
        possible_param_combination = create_grid_search_ot(param_ot)
        max_iteration = len(possible_param_combination)
        ic(len(possible_param_combination))

    param_train = dict([('reg_e', 0), ('reg_cl', 0)])
    time_start = time.time()
    nb_iteration = 0
    list_results = []

    clf = xgb.XGBClassifier()
    gridHalvingSearchCV = HalvingGridSearchCV(
        estimator=clf,
        param_grid=param_model,
        cv=10,  # integer, to specify the number of folds in a (Stratified)KFold (DOC)
        scoring='average_precision'
    ).fit(X_source, y_source)
    tuned_param_model = gridHalvingSearchCV.best_params_
    ic(tuned_param_model)

    while time.time() - time_start < 3600 * duration_max and nb_iteration < max_iteration:
        np.random.seed(4896 * nb_iteration + 5272)
        if search_method == "GridSearch" and len(possible_param_combination) > 0:
            param_train['reg_e'] = possible_param_combination[nb_iteration]['reg_e']
            param_train['reg_cl'] = possible_param_combination[nb_iteration]['reg_cl']
        else:  # Random search
            param_train['reg_e'] = param_ot['reg_e'][np.random.randint(len(param_ot['reg_e']))]
            param_train['reg_cl'] = param_ot['reg_cl'][np.random.randint(len(param_ot['reg_cl']))]
        try:
            for i in range(nb_training_iteration):
                ic(param_train)

                # First adaptation
                trans_X_target = ot_adaptation(X_source, y_source, X_target, param_train, target_to_source=True)

                # Get pseudo labels
                trans_pseudo_y_target = predict_label(tuned_param_model, X_source, y_source, trans_X_target)

                # Second adaptation
                trans2_X_target = ot_adaptation(trans_X_target, trans_pseudo_y_target, X_source, param_train)

                for j in range(5):
                    ic()
                    subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(trans2_X_target,
                                                                                           trans_pseudo_y_target,
                                                                                           p=0.5)
                    # ic(subset_trans2_X_target)
                    # ic(subset_trans_pseudo_y_target)
                    y_source_pred = predict_label(tuned_param_model,
                                                  subset_trans2_X_target,
                                                  subset_trans_pseudo_y_target,
                                                  X_source)
                    precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
                    average_precision = 100 * average_precision_score(y_source, y_source_pred)

                # add results + param for this loop to the pickle
                to_save = dict(param_train)
                to_save['precision'] = precision
                to_save['average_precision'] = average_precision
                to_save['param_model'] = tuned_param_model
                list_results.append(to_save)
                # Remark: no cross validation on the model (already tuned)
        except Exception as e:
            ic()
            print("Exception in transfer_cross_validation_trg_to_src", e)
        time.sleep(1.)  # Allow us to stop the program with ctrl-C
        nb_iteration += 1
        if to_save:
            ic(nb_iteration, to_save)
        else:
            ic(nb_iteration)
    """if not os.path.exists("OT_cross_valid_results"):
        try:
            os.makedirs("OT_cross_valid_results")
        except:
            pass
    pickle_name = f"./OT_cross_valid_results/" + pickle_name
    f = gzip.open(pickle_name, "wb")
    pickle.dump(optimal_param, f)
    f.close()"""
    optimal_param = max(list_results, key=lambda val: val['average_precision'])
    return optimal_param
