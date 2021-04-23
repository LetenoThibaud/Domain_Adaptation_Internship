import csv
import sys
import os
import gzip
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from baselines import *
from optimal_transport import *
from experimental_cv import double_cross_valid


def loadCsv(path):
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


def import_hyperparameters(algo: str, filename="hyperparameters.csv"):
    """
    :param filename:
    :param algo: name of the algorithm we want the hyperparameters of
    :return: a dictionary of hyperparameters
    """
    imported_csv_content = pd.read_csv(filename, delimiter=";")
    to_return = dict()
    column = imported_csv_content[algo]
    for i in range(len(column)):
        key, value = imported_csv_content[algo][i].split(",")
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
        data, n, d = loadCsv('datasets/satimage.data')
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
        bst = xgb.train(p, dtrain, p['num_boost_round'],
                        evallist, maximize=True,
                        early_stopping_rounds=50,
                        obj=objective_AP,
                        feval=evalerror_AP,
                        verbose_eval=False)
        rankTrain = bst.predict(dtrain)
        rankTest = bst.predict(dtest)
        rankTarget = bst.predict(dtarget)
        rankClean = bst.predict(dclean)
    return (average_precision_score(ytrain, rankTrain) * 100,
            average_precision_score(ytest, rankTest) * 100,
            average_precision_score(ytarget, rankClean) * 100,
            average_precision_score(ytarget, rankTarget) * 100)


def print_pickle(filename, type=""):
    if type == "results":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        for dataset in data:
            for algo in data.get(dataset):
                results = data[dataset][algo]
                print("Dataset:", dataset, "Algo:", algo, "Train AP {:5.2f}".format(results[0]),
                      "Test AP {:5.2f}".format(results[1]),
                      "Clean AP {:5.2f}".format(results[2]),
                      "Target AP {:5.2f}".format(results[3]),
                      "Parameters:", results[4])
    elif type == "results_adapt":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        for dataset in data:
            for algo in data.get(dataset):
                results = data[dataset][algo]
                print("Dataset:", dataset, "Algo:", algo, "Train AP {:5.2f}".format(results[0]),
                      "Test AP {:5.2f}".format(results[1]),
                      "Clean AP {:5.2f}".format(results[2]),
                      "Target AP {:5.2f}".format(results[3]),
                      "Parameters:", results[4],
                      "Parameters OT:", results[5])
    else:
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print(data)


def pickle_to_latex(filename, type=""):
    if type == "results":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print("\\begin{table}[]\n\\begin{adjustbox}{max width=1.1\\textwidth,center}\n\\begin{tabular}{llllllll}",
              "\nDataset & Algorithme & Train AP & Test AP & Clean AP & ",
              "Target AP & max\_depth & num\_boost\_round\\\\")

        for dataset in data:
            for algo in data.get(dataset):
                results = data[dataset][algo]
                print(dataset.replace("%", "\\%"), "&", algo, "&", "{:5.2f}".format(results[0]), "&",
                      "{:5.2f}".format(results[1]), "&",
                      "{:5.2f}".format(results[2]), "&", "{:5.2f}".format(results[3]),
                      "&", "{:5.2f}".format(results[4]['max_depth']),
                      "&", "{:5.2f}".format(results[4]['num_boost_round']), "\\\\")
        print("""\\end{tabular}\n\\end{adjustbox}\n\\end{table}""")

    elif type == "results_adapt":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print("\\begin{table}[]\n\\begin{adjustbox}{max width=1.1\\textwidth,center}\n\\begin{tabular}{llllllllll}",
              "\nDataset & Algorithme & Train AP & Test AP & Clean AP & ",
              "Target AP & max\_depth & num\_boost\_round & reg\_e & reg\_cl \\\\")

        for dataset in data:
            for algo in data.get(dataset):
                results = data[dataset][algo]
                print(dataset.replace("%", "\\%"), "&", algo, "&", "{:5.2f}".format(results[0]), "&",
                      "{:5.2f}".format(results[1]), "&",
                      "{:5.2f}".format(results[2]), "&", "{:5.2f}".format(results[3]),
                      "&", "{:5.2f}".format(results[4]['max_depth']),
                      "&", "{:5.2f}".format(results[4]['num_boost_round']),
                      "&", "{:5.2f}".format(results[5]['reg_e']),
                      "&", "{:5.2f}".format(results[5]['reg_cl']), "\\\\")
        print("""\\end{tabular}\n\\end{adjustbox}\n\\end{table}""")
    elif type == "results_adapt2cv":
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print("\\begin{table}[]\n\\begin{adjustbox}{max width=1.1\\textwidth,center}\n\\begin{tabular}{llllllllllll}",
              "\nDataset & Algorithme & Train AP & Test AP & Clean AP & ",
              "Target AP & max\_depth & num\_boost\_round & reg\_e & reg\_cl & CV max\_depth & CV num\_boost\_round\\\\")

        for dataset in data:
            for algo in data.get(dataset):
                results = data[dataset][algo]
                print(dataset.replace("%", "\\%"), "&", algo, "&", "{:5.2f}".format(results[0]), "&",
                      "{:5.2f}".format(results[1]), "&",
                      "{:5.2f}".format(results[2]), "&", "{:5.2f}".format(results[3]),
                      "&", "{:5.2f}".format(results[4]['max_depth']),
                      "&", "{:5.2f}".format(results[4]['num_boost_round']),
                      "&", "{:5.2f}".format(results[5]['reg_e']),
                      "&", "{:5.2f}".format(results[5]['reg_cl']),
                      "&", "{:5.2f}".format(results[6]['param_cv_model']['max_depth']),
                      "&", "{:5.2f}".format(results[6]['param_cv_model']['num_boost_round']), "\\\\")
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


def main(argv, adaptation="UOT", filename="", transpose=True, algo="XGBoost"):
    """

    :param argv:
    :param adaptation: type of adaptation wanted, default : "UOT",
                        possible values : "JCPOT", "SA", "OT", "UOT", "CORAL"
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
    for dataset in ['abalone20']:  # , 'abalone17', 'satimage', 'abalone8']:  # ['abalone8']:  #

        start = time.time()
        X, y = data_recovery(dataset)
        dataset_name = dataset
        pctPos = 100 * len(y[y == 1]) / len(y)
        dataset = "{:05.2f}%".format(pctPos) + " " + dataset
        results[adaptation] = {}
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

        # Domain adaptation
        if "OT" in adaptation:
            # we define the parameters to cross valid
            possible_reg_e = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]
            possible_reg_cl = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]

            if adaptation == "UOT":
                param_to_cv = {'reg_e': possible_reg_e, 'reg_m': possible_reg_cl}
            elif adaptation == "JCPOT":
                param_to_cv = {'reg_e': possible_reg_e}
            else:  # OT
                param_to_cv = {'reg_e': possible_reg_e, 'reg_cl': possible_reg_cl}

            cross_val_result = ot_cross_validation(Xsource, ysource, Xtarget, params_model, param_to_cv,
                                                   transpose_plan=transpose, ot_type=adaptation)

            if adaptation == "UOT":
                param_transport = {'reg_e': cross_val_result['reg_e'], 'reg_m': cross_val_result['reg_m']}
            elif adaptation == "JCPOT":
                param_transport = {'reg_e': cross_val_result['reg_e']}
            else:  # OT
                param_transport = {'reg_e': cross_val_result['reg_e'], 'reg_cl': cross_val_result['reg_cl']}

            # Transport sources to Target
            if not transpose:
                if adaptation == "UOT":
                    Xsource = uot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
                elif adaptation == "JCPOT":
                    Xsource = jcpot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
                else:  # OT
                    Xsource = ot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
            # Unbalanced optimal transport targets to Source
            else:
                if adaptation == "UOT":
                    Xtarget = uot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
                elif adaptation == "JCPOT":
                    Xtarget = jcpot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
                else:  # OT
                    Xtarget = ot_adaptation(Xsource, ysource, Xtarget, param_transport, transpose)
        elif adaptation == "SA":
            # cross validation
            param_transport = {'d': sa_cross_validation(Xsource, ysource, Xtarget, params_model, transpose)}
            # no cross validation : if we do not find the optimal dimension value by cross value
            # we set the d the maximal possible value
            #  param_transport = {'d': PCA().fit(Xsource).n_components_ - 1}
            # adaptation of the sources and targets
            original_Xsource = Xsource
            Xsource, Xtarget = sa_adaptation(Xsource, Xtarget, param_transport, transpose)
            # We have to do "adapt" Xclean because we work one subspace, by doing the following, we get the subspace of
            # Xclean but it is not adapted
            _, Xclean = sa_adaptation(original_Xsource, Xclean, param_transport, transpose=False)
        elif adaptation == "CORAL":
            # original_Xsource = Xsource
            Xsource, Xtarget = sa_adaptation(Xsource, Xtarget, transpose)
            # TODO test and remove if useless
            # We have to do "adapt" Xclean because we work one subspace, by doing the following, we get the subspace of
            # Xclean but it is not adapted
            # _, Xclean = sa_adaptation(original_Xsource, Xclean, transpose=False)


        # Learning and saving parameters :
        # From the source, training and test set are created
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

        results[adaptation][algo] = (apTrain, apTest, apClean, apTarget, params_model, param_transport,
                                     time.time() - start)
        print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
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


if __name__ == '__main__':
    # configure debugging tool
    ic.configureOutput(includeContext=True)

    # main(sys.argv, adaptation=True, filename=f"./results/comparison_results_ot_T.pklz",
    #     ot_direction="ts")

    # main(sys.argv, adaptation="JCPOT", filename=f"./results/jcpot_not_transpose_test.pklz", transpose=False,
    #     algo="XGBoost")

    # WARNING MODIFICATION OF :
    # - the dataset
    # - the seed lines 362 and 432
    # - the dataset value saved in the pickle lines 350 and 445
    # for the following tests
    main(sys.argv, adaptation="UOT", filename=f"./results/abalone20_global_compare_uot.pklz", transpose=True,
         algo="XGBoost")

    """
    main(sys.argv, adaptation="JCPOT", filename=f"./results/abalone20_global_compare_jcpot.pklz", transpose=True,
         algo="XGBoost")
    main(sys.argv, adaptation="OT", filename=f"./results/abalone20_global_compare.pklz", transpose=True,
         algo="XGBoost")
    main(sys.argv, adaptation="SA", filename=f"./results/abalone20_global_compare.pklz", transpose=True,
         algo="XGBoost")"""

    # print_pickle("results/jcpot_not_transpose_test.pklz", "results_adapt")
    # print_pickle("results/jcpot_transpose_test.pklz", "results_adapt")
    # print_pickle("results/ot_transpose_test.pklz", "results_adapt")
    # print_pickle("results/sa_transpose_test.pklz", "results_adapt")
    print_pickle("results/abalone20_global_compare_uot.pklz", "results_adapt")
    print_pickle("results/abalone20_global_compare_jcpot.pklz", "results_adapt")
    print_pickle("results/abalone20_global_compare.pklz", "results_adapt")

    """main(sys.argv, adaptation="JCPOT", filename=f"./results/comparison_results_jcpot_T.pklz", transpose=True, algo="XGBoost")
    main(sys.argv, adaptation="SA", filename=f"./results/comparison_results_sa_T.pklz", transpose=True, algo="XGBoost")
    main(sys.argv, adaptation="UOT", filename=f"./results/comparison_results_uot_T.pklz", transpose=True, algo="XGBoost")"""
