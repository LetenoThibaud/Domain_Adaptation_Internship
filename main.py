from datetime import datetime
from sys import argv, maxsize
from threading import Thread
from sklearn.decomposition import PCA
from baselines import *
from optimal_transport import *
from ot_dim_reduction import ot_dimension_reduction, dimension_reduction, reverse_dimension_reduction
import sys


# np.set_printoptions(threshold=maxsize)

def toy_example(argv, adaptation="UOT", filename="", transpose=True, algo="XGBoost", rescale=False,
                cv_with_true_labels=True, nb_iteration_cv=8, reduction=False):
    """

    :param nb_iteration_cv:
    :param cv_with_true_labels:
    :param rescale:
    :param argv:
    :param adaptation: type of adaptation wanted, default : "UOT",
                        possible values : "JCPOT", "SA", "OT", "UOT", "CORAL", "reweight_UOT"
    :param filename: name of the file where results are saved
    :param transpose: default : True (the targets are projected in the Source domain),
    False (the sources are projected in the Target domain)
    :param algo: algorithm to use for the learning
    :return:
    """
    seed = 1
    if len(argv) == 2:
        seed = int(argv[1])

    results = {}

    # 'abalone20' , 'abalone17', 'satimage', 'abalone8']:  # ['abalone8']:  #
    for dataset in ['abalone20']:

        start = time.time()
        now = datetime.now()
        file_id = now.strftime("%H%M%f")
        X, y = data_recovery(dataset)
        dataset_name = dataset
        pctPos = 100 * len(y[y == 1]) / len(y)
        dataset = "{:05.2f}%".format(pctPos) + " " + dataset
        results[dataset_name] = {}
        print(dataset)
        np.random.seed(seed)
        random.seed(seed)

        if rescale:
            normalizer = get_normalizer_data(X, "Normalizer")
            X = set_nan_to_zero(X)
            X = normalize(X, normalizer, False)
        else:
            normalizer = None

        # import the tuned parameters of the model for this dataset
        params_model = import_hyperparameters(
            dataset_name, "hyperparameters_toy_dataset.csv", toy_example=True)
        param_transport = dict()

        # Split the dataset between the source and the target(s)
        Xsource, Xtarget, ysource, ytarget = train_test_split(X, y, shuffle=True,
                                                              stratify=y,
                                                              random_state=1234,
                                                              test_size=0.51)

        """X_source_1, y_source_1, X_source_2, y_source_2, = train_test_split(Xsource, ysource, shuffle=True,
                                                                           stratify=ysource,
                                                                           random_state=1234,
                                                                           test_size=0.51)
        list_X_source = [X_source_1, X_source_2]
        list_y_source = [y_source_1, y_source_2]"""

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

        reduction_plan = None
        if reduction:
            # we cannot compute a specific reduction plan for the target bc we don't have the labels
            reduction_plan, projection = ot_dimension_reduction(Xsource, ysource)
            Xsource = dimension_reduction(Xsource, projection)
            Xtarget = dimension_reduction(Xtarget, projection)
            Xclean = dimension_reduction(Xclean, projection)

        # Tune the hyperparameters of the adaptation by cross validation
        cv_filename = f"./" + "toy" + "/cross_validation_" + file_id
        param_transport, cheat_param_transport = adaptation_cross_validation(Xsource, ysource, Xtarget,
                                                                             params_model, normalizer,
                                                                             rescale=rescale, y_target=ytarget,
                                                                             cv_with_true_labels=cv_with_true_labels,
                                                                             cv_file=cv_filename,
                                                                             transpose=transpose,
                                                                             adaptation=adaptation,
                                                                             nb_training_iteration=nb_iteration_cv)

        # param_transport, cheat_param_transport = {'reg_e': 1, 'reg_m': 1}, None

        # Xsource = reverse_dimension_reduction(Xsource, Popt)
        # Xtarget = reverse_dimension_reduction(Xtarget, Popt)
        # Xclean = reverse_dimension_reduction(Xclean, Popt)

        # Domain adaptation
        Xsource, Xtarget, Xclean = adapt_domain(Xsource, ysource, Xtarget, Xclean, param_transport,
                                                transpose,
                                                adaptation)
        # Train and Learn model :

        if reduction:
            Xsource = reverse_dimension_reduction(Xsource, reduction_plan)
            Xtarget = reverse_dimension_reduction(Xtarget, reduction_plan)
            Xclean = reverse_dimension_reduction(Xclean, reduction_plan)

        # Learning and saving parameters :
        # From the source, training and test set are created
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xsource, ysource,
                                                        shuffle=True,
                                                        random_state=3456,
                                                        stratify=ysource,
                                                        test_size=0.3)
        """if rescale:
            Xtrain = normalize(Xtrain, normalizer, True)
            Xtest = normalize(Xtest, normalizer, True)
            Xtarget = normalize(Xtarget, normalizer, True)"""

        params_model = cross_validation_model(Xsource, ysource, None)

        apTrain, apTest, apClean, apTarget = applyAlgo(algo, params_model,
                                                       Xtrain, ytrain,
                                                       Xtest, ytest,
                                                       Xtarget, ytarget,
                                                       Xclean)

        results[dataset_name][adaptation] = (apTrain, apTest, apClean, apTarget, params_model, param_transport,
                                             time.time() - start)
        print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
              "Test AP {:5.2f}".format(apTest),
              "Clean AP {:5.2f}".format(apClean),
              "Target AP {:5.2f}".format(apTarget), params_model, param_transport,
              "in {:6.2f}s".format(time.time() - start))

        results = save_results(adaptation, dataset_name, algo, apTrain, apTest, apClean, apTarget, params_model,
                               param_transport, start, filename, results, cheat_param_transport)


# MODEL
def applyAlgo(algo, p, Xtrain, ytrain, Xtest, ytest, Xtarget, ytarget, Xclean):
    if algo == 'XGBoost':
        dtrain = xgb.DMatrix(Xtrain, label=ytrain)
        dtest = xgb.DMatrix(Xtest)
        dtarget = xgb.DMatrix(Xtarget)
        dclean = xgb.DMatrix(Xclean)
        evallist = [(dtrain, 'train')]
        # p = param

        bst = xgb.train(p, dtrain, p['num_round'],
                        evallist, maximize=True,
                        early_stopping_rounds=50,
                        obj=objective_AP,
                        feval=evalerror_AP,
                        verbose_eval=False)
        rankTrain = bst.predict(dtrain)
        rankTest = bst.predict(dtest)
        rankTarget = bst.predict(dtarget)
        rankClean = bst.predict(dclean)

        """predict_y = np.array(rankTarget) > 0.5
        predict_y = predict_y.astype(int)
        print(predict_y)

        print("precision", 100 * float(sum(predict_y == ytarget)) / len(predict_y))
        print("ap", average_precision_score(ytarget, rankTarget) * 100)"""
    return (average_precision_score(ytrain, rankTrain) * 100,
            average_precision_score(ytest, rankTest) * 100,
            average_precision_score(ytarget, rankClean) * 100,
            average_precision_score(ytarget, rankTarget) * 100)


def train_model(X_source, y_source, X_target, y_target, X_clean, params_model, normalizer, rescale, algo="XGBoost"):
    """if rescale:
        ic(normalizer)
        X_source = normalize(X_source, normalizer, True)
        X_target = normalize(X_target, normalizer, True)
        X_clean = normalize(X_clean, normalizer, True)"""
    # TODO rearrange to be able to choose the normalizer
    """X_source = normalizer.inverse_transform(X_source)
        X_target = normalizer.inverse_transform(X_target)
        X_clean = normalizer.inverse_transform(X_clean)"""

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_source, y_source,
                                                    shuffle=True,
                                                    stratify=y_source,
                                                    test_size=0.3,
                                                    random_state=10)

    apTrain, apTest, apClean, apTarget = applyAlgo(algo, params_model,
                                                   Xtrain, ytrain,
                                                   Xtest, ytest,
                                                   X_target, y_target,
                                                   X_clean)

    return apTrain, apTest, apClean, apTarget


# PIPELINE ADAPTATION

def adaptation_cross_validation(Xsource, ysource, Xtarget, param_model_path, normalizer, rescale,
                                y_target=None, cv_with_true_labels=False, cv_file="cv_save",
                                nb_training_iteration=8,
                                transpose=True, adaptation="OT", cluster=1):
    if "OT" in adaptation:
        # we define the parameters to cross valid
        possible_reg_e = [0.1, 1, 5, 10]
        possible_reg_cl = [0.1, 1, 5, 10, 100]
        possible_weighted_reg_m = create_grid_search_ot(
            {"0": [1, 2, 3, 5], "1": [1, 2, 3, 5]})

        if adaptation == "UOT":
            param_to_cv = {'reg_e': possible_reg_e, 'reg_m': possible_reg_cl}
        elif adaptation == "JCPOT":
            param_to_cv = {'reg_e': possible_reg_e}
        elif adaptation == "reweight_UOT":
            param_to_cv = {'reg_e': possible_reg_e,
                           'reg_m': possible_weighted_reg_m}
        elif adaptation == "JMLOT":
            param_to_cv = {'mu': [0.5, 1, 5], 'eta': [0.0001, 0.001, 0.01, 0.1], 'sigma': [0.1, 1, 10]}
        else:  # OT
            param_to_cv = {'reg_e': possible_reg_e, 'reg_cl': possible_reg_cl}

        cross_val_result, cross_val_result_cheat = ot_cross_validation(Xsource, ysource, Xtarget, param_model_path,
                                                                       param_to_cv, normalizer,
                                                                       rescale,
                                                                       filename=cv_file,
                                                                       y_target=y_target,
                                                                       cv_with_true_labels=cv_with_true_labels,
                                                                       nb_training_iteration=nb_training_iteration,
                                                                       transpose_plan=transpose, ot_type=adaptation,
                                                                       cluster=cluster)
        if adaptation == "UOT":
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_m': cross_val_result['reg_m']}
        elif adaptation == "JCPOT":
            param_transport = {'reg_e': cross_val_result['reg_e']}
        elif adaptation == "reweight_UOT":
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_m': cross_val_result['reg_m']}
        elif adaptation == "JMLOT":
            param_transport = {'mu': cross_val_result['mu'], 'eta': cross_val_result['eta'],
                               'sigma': cross_val_result['sigma']}
        else:  # OT
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_cl': cross_val_result['reg_cl']}
        return param_transport, cross_val_result_cheat
    elif adaptation == "SA":
        params_model = import_hyperparameters("XGBoost", param_model_path)
        return {'d': components_analysis_based_method_cross_validation(Xsource, ysource, Xtarget, params_model, rescale,
                                                                       normalizer, transport_type="SA",
                                                                       extended_CV=cv_with_true_labels)}, dict()
    elif adaptation == "CORAL":
        return dict(), dict()  # equivalent to null but avoid crash later
    elif adaptation == "TCA":
        print("TO DECOMMENT")
        """return {'d': components_analysis_based_method_cross_validation(Xsource, ysource, Xtarget, params_model, rescale,
                                                                       normalizer, transport_type="TCA",
                                                                       extended_CV=cv_with_true_labels)}, dict()"""


def adapt_domain(Xsource, ysource, Xtarget, Xclean, param_transport, transpose, adaptation):
    if "OT" in adaptation:
        # Transport sources to Target
        # if not transpose:
        if adaptation == "UOT":
            Xsource = uot_adaptation(
                Xsource, ysource, Xtarget, param_transport, transpose)
        elif adaptation == "JCPOT":
            Xsource = jcpot_adaptation(
                Xsource, ysource, Xtarget, param_transport, transpose)
        elif adaptation == "reweight_UOT":
            Xsource = reweighted_uot_adaptation(
                Xsource, ysource, Xtarget, param_transport, transpose)
        elif adaptation == "JMLOT":
            Xsource = jmlot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
        else:  # OT
            Xsource = ot_adaptation(
                Xsource, ysource, Xtarget, param_transport, transpose)
        # Unbalanced optimal transport targets to Source
        """else:
            if adaptation == "UOT":
                Xtarget = uot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            elif adaptation == "JCPOT":
                Xtarget = jcpot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            elif adaptation == "reweight_UOT":
                Xtarget = reweighted_uot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            else:  # OT
                Xtarget = ot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)"""
    elif adaptation == "SA":
        original_Xsource = Xsource
        Xsource, Xtarget = sa_adaptation(
            Xsource, Xtarget, param_transport, transpose)
        # We have to do "adapt" Xclean because we work one subspace, by doing the following, we get the subspace of
        # Xclean but it is not adapted
        _, Xclean = sa_adaptation(
            original_Xsource, Xclean, param_transport, transpose=False)
    elif adaptation == "CORAL":
        # original_Xsource = Xsource
        Xsource, Xtarget = coral_adaptation(Xsource, Xtarget, transpose)
    elif adaptation == "TCA":
        # original_Xsource = Xsource
        # param_transport = {'d': 2}
        original_Xtarget = Xtarget
        Xsource, Xtarget = tca_adaptation(Xsource, Xtarget, param_transport)
        Xclean, _ = tca_adaptation(Xclean, original_Xtarget, param_transport)
    return Xsource, Xtarget, Xclean


def launch_run_jcpot(dataset, source_path, target_path, hyperparameter_file, filename, algo, adaptation_method,
                     cv_with_true_labels, transpose, nb_iteration_cv, select_feature, rescale_type="", cluster=1):
    normalizer = None

    X_source_1, y_source_1, X_source_2, y_source_2, X_source_3, y_source_3, index1, index2, index3 = import_source_per_year(
        source_path,
        select_feature)

    list_X_source = [X_source_1, X_source_2, X_source_3]
    # ic(list_X_source)

    if rescale_type != "":
        X_source_1 = get_normalizer_data(X_source_1, rescale_type)
        X_source_2 = get_normalizer_data(X_source_2, rescale_type)
        X_source_3 = get_normalizer_data(X_source_3, rescale_type)

    list_X_source = [X_source_1, X_source_2, X_source_3]
    # ic(list_X_source)
    list_y_source = [y_source_1, y_source_2, y_source_3]

    # to train the model we need to have the whole X_source in one array
    X_source, y_source, _ = import_dataset(source_path, select_feature)
    X_target, y_target, _ = import_dataset(target_path, select_feature)

    if rescale_type != "":
        X_source = get_normalizer_data(X_source, rescale_type)
        X_target = get_normalizer_data(X_target, rescale_type)

    X_clean = X_target

    params_model = import_hyperparameters(algo, hyperparameter_file)
    results = {}
    start = time.time()

    now = datetime.now()
    # create a repo per day to store the results => each repo has an id composed of the day and month
    repo_id = now.strftime("%d%m")
    file_id = now.strftime("%H%M%f")
    repo_name = "results" + repo_id
    if not os.path.exists(repo_name):
        try:
            os.makedirs(repo_name)
        except:
            pass

    results[dataset] = {}
    cv_filename = f"./" + repo_name + "/cross_validation_" + file_id
    possible_reg_e = [0.1, 1, 5, 10, 50]
    param_to_cross_valid = {'reg_e': possible_reg_e}

    param_transport, param_transport_true_label = ot_cross_validation_jcpot(list_X_source, y_source, X_target,
                                                                            None, hyperparameter_file,
                                                                            param_to_cross_valid,
                                                                            filename=cv_filename, y_target=y_target,
                                                                            cv_with_true_labels=cv_with_true_labels,
                                                                            nb_training_iteration=nb_iteration_cv,
                                                                            cluster=cluster)

    # ic(list_X_source)
    # param_transport, param_transport_true_label = {'reg_e': 1}, None
    list_X_source, X_target, X_clean = adapt_domain(list_X_source, list_y_source, X_target, X_clean, param_transport,
                                                    transpose=False, adaptation="JCPOT")

    # ic(list_X_source)
    # Creation of the filename
    if filename == "":
        if not transpose:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_classic_" + rescale_type + "_" + adaptation_method + "_" + algo + file_id
        else:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_" + adaptation_method + "_" + algo + file_id

    temp_trans_X_source = np.append(list_X_source[0], list_X_source[1], axis=0)
    temp_trans_X_source = np.append(temp_trans_X_source, list_X_source[2], axis=0)
    X_source = temp_trans_X_source
    # ic(X_source)
    # X_source = np.array(X_source)

    # ic(len(X_source), len(y_source))
    params_model = cross_validation_model(X_source, y_source, hyperparameter_file, nbFoldValid=3)

    apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, params_model,
                                                     normalizer, False, algo)
    save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, params_model,
                 param_transport, start, filename, results, param_transport_true_label)


def launch_run(dataset, source_path, target_path, hyperparameter_file, filename="", algo="XGBoost",
               adaptation_method="UOT", cv_with_true_labels=True, transpose=False, nb_iteration_cv=8,
               select_feature=False, rescale=False, reduction=False, rescale_type="", cluster=1):
    """
    :param reduction:
    :param rescale:
    :param dataset: name of the dataset
    :param source_path: path to the cvs file containing the source dataset
    :param target_path: path to the cvs file containing the target dataset
    :param hyperparameter_file: path to the cvs file containing the hyperparameters of the model
    :param filename: name of the file where the results are exported, if "" a name is generated with the name of the
                    dataset, the model, the adaptation method and a unique id based on the launch time
    :param algo: learning model
    :param adaptation_method: adaptation technique : "UOT", "OT", "JCPOT", "reweight_UOT", "TCA", "SA", "CORAL", "NA"
    :param cv_with_true_labels: boolean, if True do the cross validation of the optimal transport
                                        using the true label of the target if available
    :param transpose: boolean, if True, transport the target examples in the Source domain
    :param nb_iteration_cv: nb of iteration to use in the cross validation of the adaptation
    :param select_feature:
    :param nan_fill_strat:
    :param nan_fill_constant:
    :param n_neighbors:
    :return:
    """

    if adaptation_method != "JCPOT":
        # X_source, y_source, weights = import_dataset(source_path, select_feature,
        # rate_path="./codes/dicocluster.pickle", cluster=12) X_target, y_target, _ = import_dataset(target_path,
        # select_feature, rate_path="./codes/dicocluster.pickle", cluster=12)

        X_source, y_source, _ = import_dataset(source_path, select_feature)
        X_target, y_target, _ = import_dataset(target_path, select_feature)
        # ic(X_source, X_target)

        if rescale_type != "":
            X_source = get_normalizer_data(X_source, rescale_type)
            X_target = get_normalizer_data(X_target, rescale_type)
        normalizer = None

        # X_source, X_target = apply_PCA(X_source, X_target)

        X_clean = X_target

        reduction_plan = None
        if reduction:
            # we cannot compute a specific reduction plan for the target bc we don't have the labels
            reduction_plan, projection = ot_dimension_reduction(X_source, y_source)
            X_source = dimension_reduction(X_source, projection)
            X_target = dimension_reduction(X_target, projection)
            X_clean = dimension_reduction(X_clean, projection)

        params_model = import_hyperparameters(algo, hyperparameter_file)
        results = {}
        start = time.time()

        now = datetime.now()
        # create a repo per day to store the results => each repo has an id composed of the day and month
        repo_id = now.strftime("%d%m")
        file_id = now.strftime("%H%M%f")
        repo_name = "results" + repo_id
        if not os.path.exists(repo_name):
            try:
                os.makedirs(repo_name)
            except:
                pass

        results[dataset] = {}
        param_transport_true_label = {}
        if adaptation_method != "NA":
            cv_filename = f"./" + repo_name + "/cross_validation_" + file_id
            print(adaptation_method)
            param_transport, param_transport_true_label = adaptation_cross_validation(X_source, y_source, X_target,
                                                                                      hyperparameter_file, normalizer,
                                                                                      rescale=rescale,
                                                                                      y_target=y_target,
                                                                                      cv_with_true_labels=cv_with_true_labels,
                                                                                      cv_file=cv_filename,
                                                                                      transpose=transpose,
                                                                                      adaptation=adaptation_method,
                                                                                      nb_training_iteration=nb_iteration_cv,
                                                                                      cluster=cluster)

            # param_transport, param_transport_true_label = {'reg_e': 10, 'reg_cl': 0.1}, None
            # save_csv(X_target, "./results2005/target_after_CORAL_reg_e_10.csv")
            # save_csv(X_source, "./results2005/source_after_CORAL_reg_e_10.csv")
            X_source, X_target, X_clean = adapt_domain(X_source, y_source, X_target, X_clean, param_transport,
                                                       transpose, adaptation_method)

            sys.exit(0)
        else:
            param_transport = {}  # for the pickle

        # Creation of the filename
        if filename == "":
            if rescale:
                filename = f"./" + repo_name + "/" + dataset + \
                           "_rescale_" + rescale_type + "_" + adaptation_method + "_" + algo + file_id
            else:
                filename = f"./" + repo_name + "/" + dataset + \
                           "_" + adaptation_method + "_" + algo + file_id

        if reduction:
            X_source = reverse_dimension_reduction(X_source, reduction_plan)
            X_target = reverse_dimension_reduction(X_target, reduction_plan)
            X_clean = reverse_dimension_reduction(X_clean, reduction_plan)

        """if weights is not None:
            X_source = reverse_reweight_by_deterioration_score(X_source, weights)
            X_target = reverse_reweight_by_deterioration_score(X_target, weights)
            X_clean = reverse_reweight_by_deterioration_score(X_clean, weights)"""

        tuned_params = cross_validation_model(X_source, y_source, hyperparameter_file, nbFoldValid=3)

        apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, tuned_params,
                                                         normalizer, False, algo)

        results = save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, tuned_params,
                               param_transport, start, filename, results, param_transport_true_label)
    else:
        launch_run_jcpot(dataset, source_path, target_path, hyperparameter_file, filename, algo,
                         adaptation_method, cv_with_true_labels, transpose, nb_iteration_cv,
                         select_feature=False, rescale_type=rescale_type, cluster=cluster)


# in the main function, the thread are launched as follow :launch_thread(args).start()
def launch_thread(dataset, source_path, target_path, hyperparameter_file, filename="", algo="XGBoost",
                  adaptation_method="UOT", cv_with_true_labels=False, transpose=True, nb_iteration_cv=8,
                  select_feature=True, nan_fill_strat='mean', nan_fill_constant=0, n_neighbors=20, rescale=True):
    def handle():
        print("Thread is launch for dataset", dataset, "with algorithm", algo, "and adaptation", adaptation_method)

        """launch_run(dataset, source_path, target_path, hyperparameter_file, filename, algo,
                   adaptation_method, cv_with_true_labels, transpose, nb_iteration_cv,
                   select_feature, nan_fill_strat, nan_fill_constant, n_neighbors, rescale)"""

    t = Thread(target=handle)
    return t


def start_evaluation(clust1: int, clust2: int, adaptation=None, rescale=False):
    for i in range(clust1, clust2):
        start_evaluation_cluster(i, adaptation, rescale)


def start_evaluation_cluster(i: int, adaptation=None, transpose=False, filename="", rescale=False,
                             rescale_type=""):
    model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(
        i) + "_fraude2_best_model_and_params.csv"
    source = "./datasets_fraude2/source_" + str(i) + "_fraude2.csv"
    target = "./datasets_fraude2/target_" + str(i) + "_fraude2.csv"

    if adaptation == None:
        adaptation_methods = ["JCPOT"]
    else:
        if type(adaptation) == str:
            adaptation_methods = [adaptation]
        else:
            adaptation_methods = adaptation

    for adaptation_method in adaptation_methods:
        print("Start evaluation on cluster", i, "with adaptation", adaptation_method)
        name = "cluster" + str(i) + "_fraude2"
        launch_run(name, source, target, model_hyperparams, adaptation_method=adaptation_method,
                   nb_iteration_cv=2, transpose=transpose, cv_with_true_labels=True, filename=filename,
                   reduction=False, rescale=rescale, rescale_type=rescale_type, cluster=i)


def start_evaluation_minor_recette(i: int, adaptation=None, transpose=False, filename="", rescale=False,
                                   rescale_type="", degradation=True, select_feature=False):
    if degradation:
        suffix = "deg"
    else:
        suffix = "no_deg"

    model_hyperparams = "./hyperparameters/cluster1_fraude2_best_params_results_AP_cv.csv"
    source = "./datasets_minor_rec/source_" + str(i) + "_fraude2_" + suffix + ".csv"
    target = "./datasets_minor_rec/target_" + str(i) + "_fraude2_" + suffix + ".csv"

    if rescale:
        suffix += "_rescale"

    if adaptation == None:
        adaptation_methods = ["JCPOT"]
    else:
        if type(adaptation) == str:
            adaptation_methods = [adaptation]
        else:
            adaptation_methods = adaptation

    for adaptation_method in adaptation_methods:
        print("Start evaluation on cluster", i, "with adaptation", adaptation_method)
        name = "cluster" + str(i) + "_fraude2_"
        # filename = f"./results0206/expe_2019_no_deterioration/" + name + adaptation_method + "_" + suffix
        launch_run(name, source, target, model_hyperparams, adaptation_method=adaptation_method,
                   nb_iteration_cv=2, transpose=transpose, cv_with_true_labels=True, filename=filename,
                   reduction=False, rescale=rescale, rescale_type=rescale_type, cluster=-1,
                   select_feature=select_feature)


def apply_PCA(X_source, X_target):
    pca = PCA(n_components=30)
    pca.fit(X_target)
    return pca.transform(X_source), pca.transform(X_target)


def expe_reduction(reduct=True):
    i = 1
    dataset = "fraude2"
    model_hyperparams = "./hyperparameters/cluster1_fraude2_best_params_results_AP_cv.csv"
    source = "./datasets_minor_rec/source_" + str(i) + "_fraude2_deg.csv"
    target = "./datasets_minor_rec/target_" + str(i) + "_fraude2_deg.csv"
    algo = "XGBoost"
    adaptation_method = "WDA"
    param_transport = {}
    param_transport_true_label = {}
    results = {}
    start = time.time()
    filename = f"./results0206/expe_2019_no_deterioration/fraude2_cluster1_deg_" + adaptation_method

    X_source, y_source, _ = import_dataset(source, False)
    X_target, y_target, _ = import_dataset(target, False)

    X_source = get_normalizer_data(X_source, "Min_Max")
    X_target = get_normalizer_data(X_target, "Min_Max")

    X_clean = X_target

    if reduct:
        _, projection = ot_dimension_reduction(X_source, y_source)
        X_source = dimension_reduction(X_source, projection)
        X_target = dimension_reduction(X_target, projection)
        X_clean = dimension_reduction(X_clean, projection)

    # sysexit(0)

    tuned_params = cross_validation_model(X_source, y_source, model_hyperparams, nbFoldValid=2)
    apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, tuned_params,
                                                     None, False, algo)

    # results = save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, tuned_params,
    #                           param_transport, start, filename, results, param_transport_true_label)


"""def expe_ot_mapping_linear(degradation = True):
    if degradation:
        filename = filename = f"./results0206/expe_2019_no_deterioration/fraude2_cluster1_deg_mapping_linear"
        suffix = "deg"
    else :
        filename = filename = f"./results0206/expe_2019_no_deterioration/fraude2_cluster1_no_deg_mapping_linear"
        suffix = "no_deg"

    # Import data
    i = 1
    model_hyperparams = "./hyperparameters/cluster1_fraude2_best_params_results_AP_cv.csv"
    source = "./datasets_minor_rec/source_" + str(i) + "_fraude2_" + suffix + ".csv"
    target = "./datasets_minor_rec/target_" + str(i) + "_fraude2_" +  suffix + ".csv"
    Xsource, ysource, _ = import_dataset(source, False)
    Xtarget, ytarget, _ = import_dataset(target, False)
    Xclean = Xtarget

    start = time.time()

    # Transport source
    # Ae, be = ot.da.OT_mapping_linear(Xsource, Xtarget)
    # trXsource = Xsource.dot(Ae) + be

    mapping = ot.da.LinearTransport()
    mapping.fit(Xs=Xsource, Xt=Xtarget) 
    trXsource = mapping.transform(Xs=Xsource)

    # Learn model
    tuned_params = cross_validation_model(trXsource, ysource, model_hyperparams, nbFoldValid=3)
    apTrain, apTest, apClean, apTarget = train_model(trXsource, ysource, Xtarget, ytarget, Xclean, tuned_params,
                                                     None, False, "XGBoost")
    save_results("MapLin", "cluster1", "XGBoost", apTrain, apTest, apClean, apTarget, tuned_params,
                {}, start, filename, {}, {})"""


def expe_ot_joint_mapping_linear(degradation=True, cv_ot=False):
    if degradation:
        suffix = "deg"
        filename = filename = f"./results0206/expe_2019_no_deterioration/fraude2_cluster1_deg_joint_mapping_linear"
    else:
        suffix = "no_deg"
        filename = filename = f"./results0206/expe_2019_no_deterioration/fraude2_cluster1_no_deg_joint_mapping_linear"

    # Import data
    i = 1
    model_hyperparams = "./hyperparameters/cluster1_fraude2_best_params_results_AP_cv.csv"
    source = "./datasets_minor_rec/source_" + str(i) + "_fraude2_" + suffix + ".csv"
    target = "./datasets_minor_rec/target_" + str(i) + "_fraude2_" + suffix + ".csv"
    Xsource, ysource, _ = import_dataset(source, False)
    Xtarget, ytarget, _ = import_dataset(target, False)
    Xclean = Xtarget

    start = time.time()

    # Transport source
    # Ae, be = ot.da.OT_mapping_linear(Xsource, Xtarget)
    # trXsource = Xsource.dot(Ae) + be

    if cv_ot:
        # param_to_cv = {'mu': [0.5, 1, 5], 'eta': [0.0001, 0.001, 0.01, 0.1], 'sigma':[0.1, 1, 10]}
        param_to_cv = {'mu': [1], 'eta': [0.1], 'sigma': [0.1]}
        param_ot, cross_val_result_cheat = ot_cross_validation(Xsource, ysource, Xtarget, model_hyperparams,
                                                               param_to_cv, None,
                                                               rescale=False,
                                                               filename="cv_JMLOT",
                                                               y_target=ytarget,
                                                               cv_with_true_labels=True,
                                                               nb_training_iteration=2,
                                                               transpose_plan=False, ot_type="JMLOT",
                                                               cluster=-1)
    else:
        param_ot = {'mu': 1, 'eta': 0.001, 'sigma': 1}
        cross_val_result_cheat = {}
    trXsource = jmlot_adaptation(Xsource, ysource, Xtarget, param_ot, transpose=False)

    ic(Xsource)
    ic(Xtarget)
    ic(trXsource)

    # Learn model
    tuned_params = cross_validation_model(trXsource, ysource, model_hyperparams, nbFoldValid=2)

    # tuned_params = import_hyperparameters("XGBoost", model_hyperparams)

    apTrain, apTest, apClean, apTarget = train_model(trXsource, ysource, Xtarget, ytarget, Xclean, tuned_params,
                                                     None, False, "XGBoost")

    print(apTrain, apTest, apClean, apTarget)

    results = [{"apTrain": apTrain}, {"apTest": apTest}, {"apClean": apClean}, {"apTarget": apTarget},
               {"tuned_params": tuned_params}, {"param_ot": param_ot},
               {"cross_val_result_cheat": cross_val_result_cheat}]
    f = gzip.open(filename, "wb")
    pickle.dump(results, f)
    f.close()
    return results


if __name__ == '__main__':
    # configure debugging tool
    ic.configureOutput(includeContext=True)

    if len(argv) > 1:
        name = argv[1] + ".out"
        if argv[1] == "-wda":
            with open(name, 'w') as f_:
                sys.stdout = f_
                expe_reduction()
        elif argv[1] == "-nd_sa":
            with open(name, 'w') as f_:
                sys.stdout = f_
                start_evaluation_minor_recette(1, "SA", transpose=False, rescale=False, rescale_type="",
                                               degradation=False)
        elif argv[1] == "-jmlot_reduc_ft":
            with open(name, 'w') as f_:
                sys.stdout = f_
                start_evaluation_minor_recette(1, "JMLOT", transpose=False, rescale=False, rescale_type="",
                                               degradation=False, select_feature=False)
        elif argv[1] == "-jcpot_reduc_ft":
            with open(name, 'w') as f_:
                sys.stdout = f_
                start_evaluation_minor_recette(1, "JCPOT", transpose=False, rescale=False, rescale_type="",
                                               degradation=False, select_feature=True)
    else:
        # for silhouette_score -> labels = domains : we dont use the reel labels
        pass
