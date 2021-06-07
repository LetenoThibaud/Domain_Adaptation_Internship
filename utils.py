import csv
import gzip
import os
import pathlib
import pickle
import random
import time
import xgboost as xgb
import numpy as np
import pandas as pd
from icecream import ic
import itertools
from sklearn import preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import average_precision_score

# Create grid of parameters given parameters ranges
from sklearn.model_selection import StratifiedKFold, train_test_split


# take a grid of parameters in input and return all the possible combination (to avoid repeating the same test)
def create_grid_search_ot(params: dict):
    '''
    :param params: a dictionary containing the name of the parameters as keys and an array of their possible values
                    as values
    :return: all the possible combination of the values
    '''
    list_keys = list(params.keys())
    list_values = params.values()
    possible_combination_values = list(itertools.product(*list_values))

    possible_combination = []
    for values in possible_combination_values:
        temp_dico = dict()
        for i in range(len(list_keys)):
            key = list_keys[i]
            temp_dico[key] = values[i]
        possible_combination.append(temp_dico)

    return possible_combination


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


def cross_validation_model(X, y, hyperparameter_file=None, filename="tuned_hyperparameters.csv", algo="XGBoost",
                           export=False, nbFoldValid=4):
    if hyperparameter_file is None:
        listParams = {
            "XGBoost": listP(
                {'max_depth': range(1, 5),
                 'eta': [10 ** (-i) for i in range(1, 5)],
                 # 'subsample': np.arange(0.1, 1, 0.1),
                 # 'colsample_bytree': np.arange(0.1, 1, 0.1),
                 # 'gamma': range(0, 21),
                 'num_round': range(100, 1001, 100)
                 })
        }
    else:
        pre_tuned_params = import_hyperparameters(algo, hyperparameter_file)
        listParams = {
            "XGBoost": listP(
                {'max_depth': range(1, 5),
                 'eta': [10 ** (-i) for i in range(1, 5)],
                 'subsample': [pre_tuned_params['subsample']],
                 'colsample_bytree': [pre_tuned_params['colsample_bytree']],
                 # 'gamma': range(0, 10, 2),
                 'gamma': [pre_tuned_params['gamma']],
                 'num_round': [pre_tuned_params['num_round']]
                 })
        }

    seed = 1

    results = {}
    np.random.seed(seed)
    random.seed(seed)

    # From the source, training and test set are created
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                    shuffle=True,
                                                    stratify=y,
                                                    test_size=0.3)

    iteration = 0
    possible_param_combination = create_grid_search_ot(listParams)

    # MODEL CROSS VALIDATION
    skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
    foldsTrainValid = list(skf.split(Xtrain, ytrain))
    for algo in listParams.keys():
        start = time.time()
        validParam = []
        for param in listParams[algo]:
            valid = []
            print("Model CV combination", iteration + 1, "on", len(possible_param_combination))
            for iFoldVal in range(nbFoldValid):
                fTrain, fValid = foldsTrainValid[iFoldVal]

                dtrain = xgb.DMatrix(Xtrain[fTrain], label=ytrain[fTrain])
                dtest = xgb.DMatrix(Xtest[fValid])
                evallist = [(dtrain, 'train')]

                bst = xgb.train(param, dtrain, param['num_round'],
                                evallist, maximize=True,
                                early_stopping_rounds=50,
                                obj=objective_AP,
                                feval=evalerror_AP,
                                verbose_eval=False)

                rankTrain = bst.predict(dtrain)
                rankTest = bst.predict(dtest)

                ap_train = average_precision_score(ytrain[fTrain], rankTrain) * 100
                ap_test = average_precision_score(ytest[fValid], rankTest) * 100
                valid.append(ap_test)  # we store the ap of the test dataset for each fold of the cv
            validParam.append(np.mean(valid))
            iteration += 1
        param = listParams[algo][np.argmax(validParam)]
        if export:
            pass
            # export_hyperparameters(dataset_name, param, filename)

        return param


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


# PREPROCESSING

def filter_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1
    return data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))]


def import_source_per_year(filename, select_feature=True):
    # .drop('index', axis='columns')
    data = pd.read_csv(filename, index_col=False)

    if select_feature:
        data = feature_selection(data)

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


def import_dataset(filename, select_feature=True, rate_path=None, cluster=-1):
    data = pd.read_csv(filename, index_col=False).drop('index', axis='columns')

    if select_feature:
        data = feature_selection(data)

    weights = None
    y = data.loc[:, 'y'].to_numpy()
    if rate_path is not None and cluster != -1:
        X, weights = reweight_by_deterioration_score(data.loc[:, data.columns != 'y'], rate_path, cluster)
    else:
        X = data.loc[:, data.columns != 'y'].to_numpy()

    # data = pd.read_csv(filename, index_col=False).drop('index', axis='columns')
    # data.columns = range(data.shape[1])

    # y = data.loc[:, len(data.columns)-1]
    # X = data.loc[:, 0:len(data.columns)-2]

    X = set_nan_to_zero(X)
    # X = fill_nan(X, strategy='knn', n_neighbors=20)
    return X, y, weights


# WARNING REDUNDANT WITH def import_degradation_coeff(path, cluster: int): IN mod_cla_light_intern.py TODO : clean code
def get_degradation_rates(path, cluster):
    with open(path, 'rb') as file:
        full_dico = pickle.load(file)
        dico = dict()
        for key in full_dico.keys():
            if key[:len(str(cluster)) + 1] == (str(cluster) + ':'):
                dico[key[len(str(cluster)) + 1:]] = full_dico[key]
    return dico


def reweight_by_deterioration_score(dataframe, rate_path, cluster):
    # dictionnary containing the key(feature name) and the degradation rate applied
    rates = get_degradation_rates(rate_path, cluster)
    reweighted_X = np.array([])
    weigths = []
    for column_name in dataframe.columns:
        preprocessing.StandardScaler().fit_transform(dataframe[column_name])
        dataframe[column_name] = dataframe[column_name] * np.abs(1 - rates[column_name])
        weigths = np.append(weigths, np.abs(1 - rates[column_name]))
    return dataframe.to_numpy(), weigths


def reverse_reweight_by_deterioration_score(X, rates):
    dataframe = pd.DataFrame(X)
    for i in len(dataframe.columns):
        dataframe.iloc[:, i] = dataframe.iloc[:, i] / rates[i]
        dataframe.iloc[:, i] = preprocessing.StandardScaler().inverse_transform(dataframe.iloc[:, i])
    return dataframe.to_numpy()


def feature_selection(dataframe):
    for column_name in dataframe.columns:
        if "rto" in column_name or "ecart" in column_name or "elast" in column_name:
            dataframe = dataframe.drop(column_name, axis='columns')
    return dataframe


def get_normalizer_data(data, type):
    if type == "Standard":
        return preprocessing.StandardScaler().fit_transform(data)
    elif type == "Normalizer":
        normalizer = get_normalizer(data)
        return normalizer
    elif type == "Outliers_Robust":
        return preprocessing.RobustScaler().fit_transform(data)
    elif type == "Min_Max":
        return preprocessing.MinMaxScaler().fit_transform(data)


def get_normalizer(X, norm='l2'):
    if norm == 'l1':
        normalizer = np.abs(X).sum(axis=1)
    else:
        normalizer = np.einsum('ij,ij->i', X, X)
        np.sqrt(normalizer, normalizer)
    return normalizer


# set_nan_to_zero must be used using the name of the features => must be called during import_dataset
# (before transformation of the dataset to numpy)
def set_nan_to_zero(arr):
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=1e-3)
    imputer.fit(arr)
    arr = imputer.transform(arr)
    # to avoid true divide by 0
    arr = np.where(arr == 0, 1e-3, arr)
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


# DATA MANAGEMENT

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


# AFFICHAGE

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
    """source = normalize(source, normalizer, False)
    copy_source = normalize(source, normalizer, True)
    target = normalize(target, normalizer, False)
    copy_target = normalize(target, normalizer, True)
    ic(copy_source)
    ic(copy_target)"""

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


"""def expe_reduction():
    name = "fraude2"
    model_hyperparams = "./hyperparameters/cluster1_fraude2_best_model_and_params.csv"
    source = "./datasets_fraude2/source_1_fraude2.csv"
    target = "./datasets_fraude2/target_1_fraude2.csv"

    X_source, y_source, weights = import_dataset(source, False)
    X_target, y_target, _ = import_dataset(target, False)

    ic(len(X_source))
    ic(len(X_target))"""


def save_csv(arr, name):
    df = pd.DataFrame(arr)
    df.to_csv(name)


# Metrics
def normalize(X, normalizer, inverse):
    if not inverse:
        # TODO for i in range(X.shape[1]):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / normalizer[i]
    else:
        # TODO for i in range(X.shape[1]):
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] * normalizer[i]
    return X


def objective_AP(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    dsig = preds * (1 - preds)
    sum_pos = np.sum(preds[labels == 1])
    sum_neg = np.sum(preds[labels != 1])
    sum_tot = sum(preds)
    grad = ((labels == 1) * (-1) * sum_neg * dsig +
            (labels != 1) * dsig * sum_pos) / (sum_tot)
    hess = np.ones(len(preds)) * 0.1
    return grad, hess


def evalerror_AP(preds, dtrain):
    labels = dtrain.get_label()
    return 'AP', average_precision_score(labels, preds)


def predict_label(param, X_train, y_train, X_eval, algo='XGBoost'):
    if algo == 'XGBoost':
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_eval = xgb.DMatrix(X_eval)

        evallist = [(d_train, 'train')]
        bst = xgb.train(param, d_train, param['num_round'],
                        evallist, maximize=True,
                        early_stopping_rounds=50,
                        obj=objective_AP,
                        feval=evalerror_AP,
                        verbose_eval=False)
        prediction = bst.predict(d_eval)

        """train_prediction = bst.predict(d_train)
        threshold = round(len(y_train[y_train == 1]) / 2)
        train_prediction = np.flip(np.sort(train_prediction))
        threshold_value = train_prediction[threshold]
        labels = np.array(prediction) > threshold_value
        labels = labels.astype(int)"""

        # we compute the threshold index using the training dataset because
        # we don't have the true labels on the test one -> it is an estimation
        threshold = round(len(y_train[y_train == 1]) / 2)

        test_threshold = round(threshold * X_eval.shape[0] / X_train.shape[0])

        # with the estimation of the threshold we get the threshold value
        sorted_prediction = np.flip(np.sort(prediction))
        threshold_value = sorted_prediction[test_threshold]

        labels = np.array(prediction) > threshold_value

        labels = labels.astype(int)

        # ic(prediction)
        # ic(labels)

        return labels


def get_xgboost_model(param, X_train, y_train):
    d_train = xgb.DMatrix(X_train, label=y_train)
    evallist = [(d_train, 'train')]
    bst = xgb.train(param, d_train, param['num_round'],
                    evallist, maximize=True,
                    early_stopping_rounds=50,
                    obj=objective_AP,
                    feval=evalerror_AP,
                    verbose_eval=False)
    return bst


def predict_label_with_xgboost(model, X_train, y_train, X_eval):
    d_eval = xgb.DMatrix(X_eval)
    d_train = xgb.DMatrix(X_train, label=y_train)
    prediction = model.predict(d_eval)

    train_prediction = model.predict(d_train)
    threshold = round(len(y_train[y_train == 1]) / 4)
    train_prediction = np.flip(np.sort(train_prediction))
    threshold_value = train_prediction[threshold]

    labels = np.array(prediction) > 0.5
    labels = labels.astype(int)
    return labels
