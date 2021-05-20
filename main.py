import csv
import gzip
import pickle
import os, sys
import pathlib
import random
import time
from datetime import datetime
from sys import argv
from threading import Thread

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.impute import SimpleImputer, KNNImputer

from baselines import *
from optimal_transport import evalerror_AP, objective_AP, ot_cross_validation, uot_adaptation, jcpot_adaptation, \
    reweighted_uot_adaptation, ot_adaptation, normalize, create_grid_search_ot
from ot_dim_reduction import reverse_dimension_reduction, ot_dimension_reduction, dimension_reduction

# np.set_printoptions(threshold=sys.maxsize)

# from OTDR import ot_dimension_reduction, reverse_dimension_reduction


def import_source_per_year(filename, select_feature=True):
    # .drop('index', axis='columns')
    data = pd.read_csv(filename, index_col=False)

    source1 = data[data['index'].str.contains("_2016")]
    source2 = data[data['index'].str.contains("_2017")]
    source3 = data[data['index'].str.contains("_2018")]

    index1 = source1.loc[:, 'index'].to_numpy()
    index2 = source2.loc[:, 'index'].to_numpy()
    index3 = source3.loc[:, 'index'].to_numpy()

    source1 = source1.drop('index', axis='columns')
    source2 = source2.drop('index', axis='columns')
    source3 = source3.drop('index', axis='columns')

    if select_feature:
        source1 = feature_selection(source1)
        source2 = feature_selection(source2)
        source3 = feature_selection(source3)

    y_1 = source1.loc[:, 'y'].to_numpy()
    X_1 = source1.loc[:, source1.columns != 'y'].to_numpy()
    y_2 = source2.loc[:, 'y'].to_numpy()
    X_2 = source2.loc[:, source2.columns != 'y'].to_numpy()
    y_3 = source3.loc[:, 'y'].to_numpy()
    X_3 = source3.loc[:, source3.columns != 'y'].to_numpy()

    X_1 = set_nan_to_zero(X_1)
    X_2 = set_nan_to_zero(X_2)
    X_3 = set_nan_to_zero(X_3)

    return X_1, y_1, X_2, y_2, X_3, y_3, index1, index2, index3


def import_dataset(filename, select_feature=True):
    data = pd.read_csv(filename, index_col=False).drop('index', axis='columns')

    if select_feature:
        data = feature_selection(data)

    y = data.loc[:, 'y'].to_numpy()
    X = data.loc[:, data.columns != 'y'].to_numpy()

    # data = pd.read_csv(filename, index_col=False).drop('index', axis='columns')
    # data.columns = range(data.shape[1])

    # y = data.loc[:, len(data.columns)-1]
    # X = data.loc[:, 0:len(data.columns)-2]

    # X = set_nan_to_zero(X)
    X = fill_nan(X, strategy='knn', n_neighbors=20)
    return X, y


def feature_selection(dataframe):
    for column_name in dataframe.columns:
        if "rto" in column_name or "ecart" in column_name or "elast" in column_name:
            dataframe = dataframe.drop(column_name, axis='columns')
    return dataframe


def get_normalizer_data(data, type):
    if type == "Standard":
        return preprocessing.StandardScaler().fit(data)
    elif type == "Normalizer":
        normalizer = get_normalizer(data)
        return normalizer
    elif type == "Outliers_Robust":
        return preprocessing.RobustScaler().fit(data)


def get_normalizer(X, norm='l1'):
    if norm == 'l1':
        normalizer = np.abs(X).sum(axis=1)
    else:
        normalizer = np.einsum('ij,ij->i', X, X)
        np.sqrt(normalizer, normalizer)
    return normalizer


def reweight_preprocessing(X, coefficients):
    """
    reweight the data depending on how much impacted they are by the degradation
    :param X: dataset without nan values
    :param coefficients: array of the degradation coefficients
    :return: reweighted dataset
    """
    reweighted_X = np.array([])
    for i in range(X.shape[1]):
        weight = 1 - coefficients[i] + 0.1
        reweighted_X = np.append(reweighted_X, weight * X[i])
    return reweighted_X


# set_nan_to_zero must be used using the name of the features => must be called during import_dataset
# (before transformation of the dataset to numpy)
def set_nan_to_zero(arr):
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=1e-5)
    imputer.fit(arr)
    arr = imputer.transform(arr)
    # to avoid true divide by 0
    arr = np.where(arr == 0, 1e-5, arr)
    return arr


def fill_nan(arr, strategy='mean', fill_value=0, n_neighbors=5):
    """
    Replace NaN values in arrays (to be used notably for PCA)
    :param arr: array containing NaN values
    :param strategy: strategy to use to replace the values, can be 'mean', "median", "most_frequent" or "constant"
    :param fill_value: value used if constant is chosen
    :return:
    """
    if strategy == "knn":
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors)
    else:
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    imputer.fit(arr)
    return imputer.transform(arr)


def load_csv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    n = 0
    d = 0
    try:
        (n, d) = data.shape
    except ValueError:
        pass
    return data, n, d


def parse_value_from_cvs(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # the value is a string


def import_hyperparameters(algo: str, filename="hyperparameters.csv", toy_example=False):
    """
    :param filename:
    :param algo: name of the algorithm we want the hyperparameters of
    :return: a dictionary of hyperparameters
    """

    imported_csv_content = pd.read_csv(filename, delimiter=";")
    if not toy_example:
        algo = imported_csv_content.keys()[0]
    to_return = dict()
    column = imported_csv_content[algo]
    for i in range(len(column)):
        key, value = imported_csv_content[algo][i].split(",")
        if key != "eval_metric":
            to_return[key] = parse_value_from_cvs(value)
    return to_return


def export_hyperparameters(algo, hyperparameters, filename="hyperparameters.csv"):
    """
    :param filename:
    :param algo: name of the algo (str)
    :param hyperparameters: a dictionary of parameters we want to save
    :return:
    """
    list_hyperparam = []
    for key in hyperparameters.keys():
        list_hyperparam.append((key + "," + str(hyperparameters[key])))
    try:
        hyperparameters_dataset = pd.read_csv(filename, delimiter=";")
        hyperparameters_dataset[algo] = list_hyperparam
    except FileNotFoundError:
        hyperparameters_dataset = pd.DataFrame(columns=[algo], data=list_hyperparam)
    hyperparameters_dataset.to_csv(filename, index=False)


def data_recovery(dataset):
    # Depending on the value on the 8th columns, the label is attributed
    # then we can have three differents datasets :
    # - abalone8
    # - abalone17
    # - abalone20
    if dataset in ['abalone8', 'abalone17', 'abalone20']:
        data = pd.read_csv("datasets/abalone.data", header=None)
        data = pd.get_dummies(data, dtype=float)
        if dataset in ['abalone8']:
            y = np.array([1 if elt == 8 else 0 for elt in data[8]])
        elif dataset in ['abalone17']:
            y = np.array([1 if elt == 17 else 0 for elt in data[8]])
        elif dataset in ['abalone20']:
            y = np.array([1 if elt == 20 else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))
    elif dataset in ['satimage']:
        data, n, d = load_csv('datasets/satimage.data')
        X = data[:, np.arange(d - 1)].astype(float)
        y = data[:, d - 1]
        y = y.astype(int)
        y[y != 4] = 0
        y[y == 4] = 1
    return X, y


# Create grid of parameters given parameters ranges
def listP(dic):
    params = list(dic.keys())
    listParam = [{params[0]: value} for value in dic[params[0]]]
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    return listParam


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


def print_whole_repo(repo, constraint=""):
    for file in pathlib.Path(repo).iterdir():
        path = str(file)
        if not "ipynb" in path:
            if len(constraint) > 1 and constraint in path:
                if "OT" in path:
                    print_pickle(path)
                    print(" ")
            elif len(constraint) <= 1:
                print_pickle(path)
                print(" ")


def latex_whole_repo(repo, constraint=""):
    for file in pathlib.Path(repo).iterdir():
        path = str(file)
        print("Data saved in", path)
        if not "ipynb" in path:
            if len(constraint) > 1 and constraint in path:
                if "OT" in path:
                    pickle_to_latex(path, "results_adapt")
                    print(" ")
            elif len(constraint) <= 1:
                pickle_to_latex(path, "results_adapt")
                print(" ")


def print_pickle(filename, type=""):
    if type == "results":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        for dataset in data:
            for transport in data.get(dataset):
                results = data[dataset][transport]
                print("Dataset:", dataset, "Transport method :", transport, "Algo:", results[0],
                      "Train AP {:5.2f}".format(results[1]),
                      "Test AP {:5.2f}".format(results[2]),
                      "Clean AP {:5.2f}".format(results[3]),
                      "Target AP {:5.2f}".format(results[4]),
                      "Parameters:", results[7])
    elif type == "results_adapt":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        for dataset in data:
            for transport in data.get(dataset):
                results = data[dataset][transport]
                print("Dataset:", dataset, "Transport method :", transport, "Algo:", results[0],
                      "Train AP {:5.2f}".format(results[1]),
                      "Test AP {:5.2f}".format(results[2]),
                      "Clean AP {:5.2f}".format(results[3]),
                      "Target AP {:5.2f}".format(results[4]),
                      "Parameters:", results[5],
                      "Parameters OT:", results[6])
    else:
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print(data)


def pickle_to_latex(filenames, type=""):
    if type == "results":
        print("\\begin{table}[]\n\\begin{adjustbox}{max width=1.1\\textwidth,center}\n\\begin{tabular}{lllllll}",
              "\nDataset & Algorithme & Transport & Train AP & Test AP & Clean AP & ",
              "Target AP\\\\")
        for filename in filenames:
            file = gzip.open(filename, 'rb')
            data = pickle.load(file)
            file.close()
            for dataset in data:
                for transport in data.get(dataset):
                    results = data[dataset][transport]
                    print(dataset.replace("%", "\\%"), "&", results[0], "&", transport, "&",
                          "{:5.2f}".format(results[1]), "&",
                          "{:5.2f}".format(results[2]), "&",
                          "{:5.2f}".format(results[3]), "&", "{:5.2f}".format(results[4]),
                          "\\\\")
        print("""\\end{tabular}\n\\end{adjustbox}\n\\end{table}""")
    elif type == "results_adapt":
        print("\\begin{table}[]\n\\begin{adjustbox}{max width=1.1\\textwidth,center}\n\\begin{tabular}{lllllllll}",
              "\nDataset & Algorithme &  Transport & Train AP & Test AP & Clean AP & ",
              "Target AP & param_OT & param_OT_true_labels \\\\")
        file = gzip.open(filenames, 'rb')
        data = pickle.load(file)
        file.close()
        for dataset in data:
            for transport in data.get(dataset):
                results = data[dataset][transport]
                print(dataset.replace("%", "\\%"), "&", results[0], "&", transport, "&",
                      "{:5.2f}".format(results[1]), "&",
                      "{:5.2f}".format(results[2]), "&",
                      "{:5.2f}".format(results[3]), "&", "{:5.2f}".format(results[4]),
                      "&", results[6],
                      "&", results[7],
                      "\\\\")
        print("""\\end{tabular}\n\\end{adjustbox}\n\\end{table}""")


def cross_validation_model(filename="tuned_hyperparameters.csv"):
    listParams = {
        "XGBoost": listP(
            {'max_depth': range(1, 6),
             # 'eta': [10 ** (-i) for i in range(1, 5)],
             # 'subsample': np.arange(0.1, 1, 0.1),
             # 'colsample_bytree': np.arange(0.1, 1, 0.1),
             'gamma': range(0, 21),
             # 'num_boost_round': range(100, 1001, 100)
             })
    }

    nbFoldValid = 5
    seed = 1

    results = {}
    for dataset in ['abalone20', 'abalone17', 'satimage', 'abalone8']:  # ['abalone8']:  #
        X, y = data_recovery(dataset)
        dataset_name = dataset
        pctPos = 100 * len(y[y == 1]) / len(y)
        dataset = "{:05.2f}%".format(pctPos) + " " + dataset
        print(dataset)
        np.random.seed(seed)
        random.seed(seed)

        Xsource, Xtarget, ysource, ytarget = train_test_split(X, y, shuffle=True,
                                                              stratify=y,
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
                Xtarget[np.random.choice(len(Xtarget), int(len(Xtarget) / 2)),
                        feat] = 0

        # From the source, training and test set are created
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xsource, ysource,
                                                        shuffle=True,
                                                        stratify=ysource,
                                                        test_size=0.3)

        # MODEL CROSS VALIDATION
        skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        foldsTrainValid = list(skf.split(Xtrain, ytrain))
        results[dataset] = {}
        for algo in listParams.keys():
            start = time.time()
            if len(listParams[algo]) > 1:  # Cross validation
                validParam = []
                for param in listParams[algo]:
                    valid = []
                    for iFoldVal in range(nbFoldValid):
                        fTrain, fValid = foldsTrainValid[iFoldVal]
                        valid.append(applyAlgo(algo, param,
                                               Xtrain[fTrain], ytrain[fTrain],
                                               Xtrain[fValid], ytrain[fValid],
                                               Xtarget, ytarget, Xclean)[1])
                    validParam.append(np.mean(valid))
                param = listParams[algo][np.argmax(validParam)]
            else:  # No cross-validation
                param = listParams[algo][0]

            # LEARNING AND SAVING PARAMETERS
            apTrain, apTest, apClean, apTarget = applyAlgo(algo, param,
                                                           Xtrain, ytrain,
                                                           Xtest, ytest,
                                                           Xtarget, ytarget,
                                                           Xclean)
            results[dataset][algo] = (apTrain, apTest, apClean, apTarget, param)
            print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
                  "Test AP {:5.2f}".format(apTest),
                  "Clean AP {:5.2f}".format(apClean),
                  "Target AP {:5.2f}".format(apTarget), param,
                  "in {:6.2f}s".format(time.time() - start))
        export_hyperparameters(dataset_name, param, filename)


def adaptation_cross_validation(Xsource, ysource, Xtarget, params_model, normalizer, rescale,
                                y_target=None, cv_with_true_labels=False,
                                nb_training_iteration=8,
                                transpose=True, adaptation="UOT"):
    if "OT" in adaptation:
        # we define the parameters to cross valid
        possible_reg_e = [0.001, 0.01, 0.1, 1, 10]
        possible_reg_cl = [0.001, 0.01, 0.1, 1, 10]
        possible_weighted_reg_m = create_grid_search_ot(
            {"0": [1, 2, 3, 5], "1": [1, 2, 3, 5]})

        if adaptation == "UOT":
            param_to_cv = {'reg_e': possible_reg_e, 'reg_m': possible_reg_cl}
        elif adaptation == "JCPOT":
            param_to_cv = {'reg_e': possible_reg_e}
        elif adaptation == "reweight_UOT":
            param_to_cv = {'reg_e': possible_reg_e,
                           'reg_m': possible_weighted_reg_m}
        else:  # OT
            param_to_cv = {'reg_e': possible_reg_e, 'reg_cl': possible_reg_cl}

        cross_val_result, cross_val_result_cheat = ot_cross_validation(Xsource, ysource, Xtarget, params_model,
                                                                       param_to_cv, normalizer,
                                                                       rescale,
                                                                       y_target=y_target,
                                                                       cv_with_true_labels=cv_with_true_labels,
                                                                       nb_training_iteration=nb_training_iteration,
                                                                       transpose_plan=transpose, ot_type=adaptation)
        if adaptation == "UOT":
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_m': cross_val_result['reg_m']}
        elif adaptation == "JCPOT":
            param_transport = {'reg_e': cross_val_result['reg_e']}
        elif adaptation == "reweight_UOT":
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_m': cross_val_result['reg_m']}
        else:  # OT
            param_transport = {
                'reg_e': cross_val_result['reg_e'], 'reg_cl': cross_val_result['reg_cl']}
        return param_transport, cross_val_result_cheat
    elif adaptation == "SA":
        return {'d': components_analysis_based_method_cross_validation(Xsource, ysource, Xtarget, params_model, rescale,
                                                                       normalizer, transport_type="SA",
                                                                       extended_CV=cv_with_true_labels)}, dict()
    elif adaptation == "CORAL":
        return dict(), dict()  # equivalent to null but avoid crash later
    elif adaptation == "TCA":
        return {'d': components_analysis_based_method_cross_validation(Xsource, ysource, Xtarget, params_model, rescale,
                                                                       normalizer, transport_type="TCA",
                                                                       extended_CV=cv_with_true_labels)}, dict()


def adapt_domain(Xsource, ysource, Xtarget, Xclean, param_transport, transpose, adaptation):
    if "OT" in adaptation:
        # Transport sources to Target
        if not transpose:
            if adaptation == "UOT":
                Xsource = uot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            elif adaptation == "JCPOT":
                Xsource = jcpot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            elif adaptation == "reweight_UOT":
                Xsource = reweighted_uot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
            else:  # OT
                Xsource = ot_adaptation(
                    Xsource, ysource, Xtarget, param_transport, transpose)
        # Unbalanced optimal transport targets to Source
        else:
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
                    Xsource, ysource, Xtarget, param_transport, transpose)
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


def train_model(X_source, y_source, X_target, y_target, X_clean, params_model, normalizer, rescale, algo="XGBoost"):
    if rescale:
        ic(normalizer)
        X_source = normalize(X_source, normalizer, True)
        X_target = normalize(X_target, normalizer, True)
        X_clean = normalize(X_clean, normalizer, True)
        # TODO rearrange to be able to choose the normalizer
        """X_source = normalizer.inverse_transform(X_source)
        X_target = normalizer.inverse_transform(X_target)
        X_clean = normalizer.inverse_transform(X_clean)"""

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_source, y_source,
                                                    shuffle=True,
                                                    stratify=y_source,
                                                    test_size=0.3)

    apTrain, apTest, apClean, apTarget = applyAlgo(algo, params_model,
                                                   Xtrain, ytrain,
                                                   Xtest, ytest,
                                                   X_target, y_target,
                                                   X_clean)

    return apTrain, apTest, apClean, apTarget


def save_results(adaptation, dataset, algo, apTrain, apTest, apClean, apTarget, params_model, param_transport, start,
                 filename, results, param_transport_true_labels=None):
    if param_transport_true_labels is None:
        param_transport_true_labels = {}
    results[dataset][adaptation] = (algo, apTrain, apTest, apClean, apTarget, params_model, param_transport,
                                    param_transport_true_labels, time.time() - start)

    print(param_transport_true_labels)
    print(dataset, algo, adaptation, "Train AP {:5.2f}".format(apTrain),
          "Test AP {:5.2f}".format(apTest),
          "Clean AP {:5.2f}".format(apClean),
          "Target AP {:5.2f}".format(apTarget), params_model, param_transport, param_transport_true_labels,
          "in {:6.2f}s".format(time.time() - start))

    if not os.path.exists("results"):
        try:
            os.makedirs("results")
        except:
            pass
    if filename == "":
        filename = f"./results/" + dataset + adaptation + algo + ".pklz"
    f = gzip.open(filename, "wb")
    pickle.dump(results, f)
    f.close()
    return results


def launch_run_jcpot(dataset, source_path, target_path, hyperparameter_file, filename, algo, adaptation_method,
                     cv_with_true_labels, transpose, nb_iteration_cv, select_feature):
    rescale = False
    normalizer = None

    X_source_1, y_source_1, X_source_2, y_source_2, X_source_3, y_source_3, index1, index2, index3 = import_source_per_year(
        source_path,
        select_feature)
    list_X_source = [X_source_1, X_source_2, X_source_3]
    list_y_source = [y_source_1, y_source_2, y_source_3]

    # to train the model we need to have the whole X_source in one array
    X_source, y_source = import_dataset(source_path, select_feature)
    ic(X_source)
    X_target, y_target = import_dataset(target_path, select_feature)

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
    """param_transport, param_transport_true_label = adaptation_cross_validation(list_X_source, list_y_source, X_target,
                                                                              params_model, normalizer,
                                                                              rescale, y_target=y_target,
                                                                              cv_with_true_labels=cv_with_true_labels,
                                                                              transpose=transpose,
                                                                              adaptation=adaptation_method,
                                                                              nb_training_iteration=nb_iteration_cv)"""

    param_transport, param_transport_true_label = {'reg_e': 1}, None

    X_source, X_target, X_clean = adapt_domain(list_X_source, list_y_source, X_target, X_clean, param_transport,
                                               transpose, adaptation_method)

    # Creation of the filename
    if filename == "":
        if not transpose:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_classic_" + adaptation_method + "_" + algo + file_id
        else:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_" + adaptation_method + "_" + algo + file_id

    list_X_source = X_source
    temp_trans_X_source = np.append(X_source[0], X_source[1], axis=0)
    temp_trans_X_source = np.append(temp_trans_X_source, X_source[2], axis=0)
    X_source = temp_trans_X_source
    # X_source = np.array(X_source)

    apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, params_model,
                                                     normalizer, rescale, algo)
    save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, params_model,
                 param_transport, start, filename, results, param_transport_true_label)

    index1 = np.reshape(index1, (-1, 1))
    index2 = np.reshape(index2, (-1, 1))
    index3 = np.reshape(index3, (-1, 1))
    ic(index1, index2, index3)
    temp_trans_X_source = np.append(np.append(list_X_source[0], index1, axis=1),
                                    np.append(list_X_source[1], index2, axis=1), axis=0)
    temp_trans_X_source = np.append(temp_trans_X_source, np.append(list_X_source[2], index3, axis=1), axis=0)
    X_source = temp_trans_X_source
    save_csv(X_source, "source_after_JCPOT_index")


def launch_run(dataset, source_path, target_path, hyperparameter_file, filename="", algo="XGBoost",
               adaptation_method="UOT", cv_with_true_labels=True, transpose=True, nb_iteration_cv=8,
               select_feature=True, nan_fill_strat='mean', nan_fill_constant=0, n_neighbors=20, rescale=False,
               reduction=False):
    """
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
        X_source, y_source = import_dataset(source_path, select_feature)
        X_target, y_target = import_dataset(target_path, select_feature)
        # ic(X_source, X_target)
        if rescale:
            normalizer = get_normalizer_data(X_source, "Normalizer")
            # ic(normalizer)
            # normalizer = get_normalizer_data(X_source, "Outliers_Robust")
            X_source = normalize(X_source, normalizer, False)
            X_target = normalize(X_target, normalizer, False)
        else:
            normalizer = None
        X_clean = X_target
        # ic(X_source, X_target, X_clean)

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
            """param_transport, param_transport_true_label = adaptation_cross_validation(X_source, y_source, X_target,
                                                                                      params_model, normalizer,
                                                                                      rescale, y_target=y_target,
                                                                                      cv_with_true_labels=cv_with_true_labels,
                                                                                      transpose=transpose,
                                                                                      adaptation=adaptation_method,
                                                                                      nb_training_iteration=nb_iteration_cv)"""

            param_transport, param_transport_true_label = {'reg_e': 10, 'reg_cl': 0.1}, None
            # save_csv(X_target, "./results2005/target_after_CORAL_reg_e_10.csv")
            # save_csv(X_source, "./results2005/source_after_CORAL_reg_e_10.csv")
            X_source, X_target, X_clean = adapt_domain(X_source, y_source, X_target, X_clean, param_transport,
                                                       transpose, adaptation_method)
        else:
            param_transport = {}  # for the pickle

        # Creation of the filename
        if filename == "":
            if not transpose:
                filename = f"./" + repo_name + "/" + dataset + \
                           "_classic_" + adaptation_method + "_" + algo + file_id
            else:
                filename = f"./" + repo_name + "/" + dataset + \
                           "_" + adaptation_method + "_" + algo + file_id
        if adaptation_method in {"SA", "CORAL", "TCA"}:
            if not select_feature:
                ic()
                if nan_fill_strat == "constant":
                    param_transport['nan_fill'] = {
                        nan_fill_strat: nan_fill_constant}
                    filename = f"./" + repo_name + "/" + dataset + "_" + adaptation_method + "_" + nan_fill_strat + "_" + \
                               algo + "_" + file_id
                elif nan_fill_strat == "knn":
                    param_transport['nan_fill'] = {nan_fill_strat: n_neighbors}
                    filename = f"./" + repo_name + "/" + dataset + "_" + adaptation_method + "_" + str(
                        n_neighbors) + "nn_" + algo + "_" + file_id
                else:
                    param_transport['nan_fill'] = nan_fill_strat
                    filename = f"./" + repo_name + "/" + dataset + "_" + adaptation_method + "_" + nan_fill_strat + "_" + \
                               algo + "_" + file_id

        if reduction:
            X_source = reverse_dimension_reduction(X_source, reduction_plan)
            X_target = reverse_dimension_reduction(X_target, reduction_plan)
            X_clean = reverse_dimension_reduction(X_clean, reduction_plan)

        apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, params_model,
                                                         normalizer, rescale, algo)
        results = save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, params_model,
                               param_transport, start, filename, results, param_transport_true_label)
    else:
        launch_run_jcpot(dataset, source_path, target_path, hyperparameter_file, filename, algo,
                         adaptation_method, cv_with_true_labels, transpose, nb_iteration_cv, rescale)


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
        """param_transport, cheat_param_transport = adaptation_cross_validation(Xsource, ysource, Xtarget,
                                                                             params_model, normalizer,
                                                                             rescale=rescale, y_target=ytarget,
                                                                             cv_with_true_labels=cv_with_true_labels,
                                                                             transpose=transpose,
                                                                             adaptation=adaptation,
                                                                             nb_training_iteration=nb_iteration_cv)"""

        param_transport, cheat_param_transport = {'reg_e': 1, 'reg_m': 1}, None

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
        if rescale:
            Xtrain = normalize(Xtrain, normalizer, True)
            Xtest = normalize(Xtest, normalizer, True)
            Xtarget = normalize(Xtarget, normalizer, True)

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


# in the main function, the thread are launched as follow :launch_thread(args).start()
def launch_thread(dataset, source_path, target_path, hyperparameter_file, filename="", algo="XGBoost",
                  adaptation_method="UOT", cv_with_true_labels=False, transpose=True, nb_iteration_cv=8,
                  select_feature=True, nan_fill_strat='mean', nan_fill_constant=0, n_neighbors=20, rescale=True):
    def handle():
        print("Thread is launch for dataset", dataset, "with algorithm", algo, "and adaptation", adaptation_method)

        launch_run(dataset, source_path, target_path, hyperparameter_file, filename, algo,
                   adaptation_method, cv_with_true_labels, transpose, nb_iteration_cv,
                   select_feature, nan_fill_strat, nan_fill_constant, n_neighbors, rescale)

    t = Thread(target=handle)
    return t


def start_evaluation(clust1: int, clust2: int, adaptation=None, rescale=False):
    for i in range(clust1, clust2):
        start_evaluation_cluster(i, adaptation, rescale)


def start_evaluation_cluster(i: int, adaptation=None, transpose=False, filename="", reduction=False):
    model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(
        i) + "_fraude2_best_model_and_params.csv"
    #

    source = "./datasets_fraude2/source_" + str(i) + "_fraude2.csv"
    target = "./datasets_fraude2/target_" + str(i) + "_fraude2.csv"

    if adaptation == None:
        adaptation_methods = [
            "JCPOT"]  # ["NA", "CORAL", "UOT", "OT", "SA", "reweight_UOT", "JCPOT"]  # ["UOT", "OT", "SA", "NA", "CORAL"] # ,"reweight_UOT", "JCPOT", "TCA"
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
                   reduction=reduction, rescale=True)


def expe_norm():
    source = np.array([(3.89769605, 5.62046383, 9.26488206, 0.09981507, 2.95457575, 4.68247455
                        , 5.43107062, 5.23233488, 8.88920099, 7.22975435, 1.25516338, 8.29336266
                        , 2.83238462, 4.35684671, 2.38896096, 8.63382591, 0.68686362, 6.32632873
                        , 4.1162196, 7.90343913, 3.41238392, 7.71851642, 6.39387483, 9.39881661
                        , 6.67785801, 1.80384883, 6.98794295, 1.69740799, 6.13168769, 9.64553036
                        , 6.6388268, 1.13082919, 5.45609879, 7.98989938, 5.62725099, 0.51951685
                        , 0.46850312, 0.49748708, 9.43152486, 9.19425823, 5.35887637, 9.19382174
                        , 0.28722491, 7.30170863, 3.32040736, 5.59300537, 7.94214332, 3.65567933
                        , 5.63458867, 9.49029065),
                       (1.61907833, 5755.21911145, 830.71754681, 393.87737453, 412.37364712
                        , 732.05128364, 497.34143205, 12.16969699, 931.81361921, 316.7469385
                        , 966.11192076, 811.65440449, 286.75893741, 276.71701864, 487.30875056
                        , 758.3039572, 672.59535521, 10535.10561324, 456.46789173, 176.91159203
                        , 177.57890776, 473.95952469, 823.21213695, 3.63945661, 284.4701334
                        , 396.90338021, 245.75559154, 540.86761559, 876.01099834, 542.34330783
                        , 741.08403127, 211.18476877, 172.42746437, 201.0806242, 433.79685765
                        , 818.55368542, 8.8846094, 693.26025479, 297.01432028, 964.11162844
                        , 15185.98579996, 710.55824143, 572.64578693, 946.36587084, 12730.85290656
                        , 701.65152737, 944.46937738, 330.25522313, 473.88410253, 65.53585207),
                       (0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
                        1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0)])

    target = np.array([0.2 * source[0], 1.5 * source[1], source[2]])

    ic(source)
    ic(target)
    normalizer = get_normalizer(source, 'l2')
    source = normalize(source, normalizer, False)
    copy_source = normalize(source, normalizer, True)
    target = normalize(target, normalizer, False)
    copy_target = normalize(target, normalizer, True)
    ic(copy_source)
    ic(copy_target)

    """Xsource = source[0:2, :]
    ysource = source[2, :]
    ic(Xsource)
    Xtarget = target[0:2, :]"""

    """Xclean = Xtarget
    param_transport = {"reg_e": 0.5, "reg_cl": 0.1}

    source, target, clean = adapt_domain(Xsource, ysource, Xtarget, Xclean, param_transport, transpose=True,
                                         adaptation="OT")

    ic(copy_source[:, 0:1])
    ic(source)"""


def expe_reduction():
    name = "fraude2"
    model_hyperparams = "./hyperparameters/cluster20_fraude2_best_model_and_params.csv",
    source = "./datasets/source_20_fraude2.csv",
    target = "./datasets/target_20_fraude2.csv",

    launch_run(name, source, target, model_hyperparams, adaptation_method="UOT",
               nb_iteration_cv=2, rescale=False, reduction=True, cv_with_true_labels=True,
               filename=f"./results1105/expe_reduction")


def save_csv(arr, name):
    df = pd.DataFrame(arr)
    df.to_csv(name)


if __name__ == '__main__':
    # configure debugging tool
    ic.configureOutput(includeContext=True)

    if len(argv) > 1:
        if argv[1] == "-launch":
            """if argv[4] == "False":
                transpose = False
            else:
                transpose = True"""
            # print("Start evaluation on cluster", int(argv[2])+1, "with adaptation",
            #      argv[3])  # , "and rescale is", rescale)
            if argv[3] == "None":
                start_evaluation_cluster(
                    int(argv[2]), "JCPOT", transpose=False)
            else:
                start_evaluation_cluster(int(argv[2]), argv[3])
        elif argv[1] == "-print_cluster":
            latex_whole_repo("./results1805/")
        elif argv[1] == "-expe_jcpot1":
            start_evaluation_cluster(12, "JCPOT", transpose=False)
        elif argv[1] == "-expe_ot":
            start_evaluation_cluster(12, "OT", transpose=True)
        elif argv[1] == "-expe_ot_red":
            start_evaluation_cluster(12, "OT", transpose=True, reduction=True)
    else:
        # start_evaluation_cluster(12, "UOT", transpose=True)
        # print_whole_repo("./results1805/")
        # print_whole_repo("./results1905/")

        # start_evaluation_cluster(12, "SA", transpose=True)

        data = pd.read_csv("source_after_JCPOT_index.csv", index_col=False)
        ic(data)
        data = data.drop(data[data.iloc[:, 3] == 0.0].index)
        open_file = open("./results2005/nb_sir.txt", 'w')
        for siren in data.iloc[:, 213]:
            open_file.write(siren + '\n')
        open_file.close()

