import os
import csv
import gzip
import pickle
import random
import sys
import ot
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
# tool to debug
from icecream import ic



def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


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


def predict_label(param, X_train, y_train, X_eval, algo='XGBoost'):
    if algo == 'XGBoost':
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_eval = xgb.DMatrix(X_eval)

        evallist = [(d_train, 'train')]
        bst = xgb.train(param, d_train, param['num_boost_round'],
                        evallist, maximize=True,
                        early_stopping_rounds=50,
                        obj=objective_AP,
                        feval=evalerror_AP,
                        verbose_eval=False)
        prediction = bst.predict(d_eval)

        labels = np.array(prediction) > 0.5
        labels = labels.astype(int)

        return labels


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


def transfer_cross_validation_trg_to_src(X_source, y_source, X_target, param_model, pickle_name, duration_max=48,
                                         nb_training_iteration=10):
    '''
    :param X_source:
    :param y_source:
    :param X_target:
    :param param_model:
    :param pickle_name:
    :param duration_max:
    :param nb_training_iteration:
    :return: dictionary containing optimal reg_e, reg_cl, precision value and average_precision value
    '''
    reg_e_loop = [0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    reg_cl_loop = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]

    param_train = dict([('reg_e', 0), ('reg_cl', 0)])
    time_start = time.time()
    nb_iteration = 0
    list_results = []
    while time.time() - time_start < 3600 * duration_max and nb_iteration < 10:
        np.random.seed(4896 * nb_iteration + 5272)
        param_train['reg_e'] = reg_e_loop[np.random.randint(len(reg_e_loop))]
        param_train['reg_cl'] = reg_cl_loop[np.random.randint(len(reg_cl_loop))]
        results = []
        try:
            for i in range(nb_training_iteration):
                ic(param_train)
                # Do the first adaptation (from source to target for the plan but adapt with the transpose)
                trans_X_target = ot_adaptation(X_source, y_source, X_target, param_train, target_to_source=True)

                # Get pseudo labels
                trans_pseudo_y_target = predict_label(param_model, X_source, y_source, trans_X_target)

                # Do the second adaptation (from target to source)
                # We don't use target_to_source = True, instead we reverse the target and source in parameters
                # bc we don't want to use the transpose of a plan here, just create a plan from Target to Source
                trans2_X_target = ot_adaptation(trans_X_target, trans_pseudo_y_target, X_source, param_train)

                # TODO Check 10 times cf code MLOT
                for j in range(10):
                    ic()
                    subset_trans2_X_target, subset_trans_pseudo_y_target = generateSubset2(trans2_X_target,
                                                                                           trans_pseudo_y_target,
                                                                                           p=0.5)
                    # ic(subset_trans2_X_target)
                    # ic(subset_trans_pseudo_y_target)
                    y_source_pred = predict_label(param_model,
                                                  subset_trans2_X_target,
                                                  subset_trans_pseudo_y_target,
                                                  X_source)
                    precision = 100 * float(sum(y_source_pred == y_source)) / len(y_source_pred)
                    average_precision = 100 * average_precision_score(y_source, y_source_pred)
                    # res = "Precision : " + str(precision) + "Average precision : " + str(average_precision)
                    # results.append(res)
                # TODO add results + param for this loop to the pickle
                to_save = dict(param_train)
                to_save['precision'] = precision
                to_save['average_precision'] = average_precision
                list_results.append(to_save)
                ic(to_save)
                ic(list_results)
                if not os.path.exists("OT_cross_valid_results"):
                    try:
                        os.makedirs("OT_cross_valid_results")
                    except:
                        pass
                pickle_name = f"./OT_cross_valid_results/" + pickle_name
                f = gzip.open(pickle_name, "wb")
                pickle.dump(to_save, f)
                f.close()
                # Remark: no cross validation on the model (already tuned)
        except Exception as e:
            ic()
            print("Exception in transfer_cross_validation_trg_to_src", e)
        time.sleep(1.)  # Allow us to stop the program with ctrl-C
        nb_iteration += 1
        ic(nb_iteration, list_results)

    optimal_param = max(list_results, key=lambda val: val['average_precision'])
    return optimal_param


def ot_adaptation(X_source, y_source, X_target, param_ot, target_to_source=False):
    """
    Function computes the transport plan and transport the sources to the targets
    or the reverse
    :param param_model:
    :param X_source: Source features
    :param y_source: Source labels
    :param X_target: Target features
    :param target_to_source: boolean set by default to False (transport sources to targets)
    if boolean is set to True the X_target is transported in the Source domain
    :return: Return the source features transported into the target if target_to_source = False
            Return the target features transported into the source if target_to_source = True
    """
    # transport = ot.da.EMDTransport()
    # doc : Domain Adaptation OT method based on sinkhorn algorithm + LpL1 class regularization.
    # LpL1 : class-label based regularizer built upon an lp − l1 norm (Courty et al., 2016)
    ''' From the doc :
    - fit(Xs=None, ys=None, Xt=None, yt=None)
                        -> Build a coupling matrix from source and target sets of samples (Xs, ys) and (Xt, yt)
    - fit_transform(Xs=None, ys=None, Xt=None, yt=None)
                        -> Build a coupling matrix from source and target sets of samples (Xs, ys) and (Xt, yt) 
                        and transports source samples Xs onto target ones Xt
    - transform(Xs=None, ys=None, Xt=None, yt=None, batch_size=128)
                        -> Transports source samples Xs onto target ones Xt
    - inverse_transform(Xs=None, ys=None, Xt=None, yt=None, batch_size=128 
                        -> Transports target samples Xt onto source samples Xs
    '''

    reg_entropy = param_ot['reg_e']
    reg_classes = param_ot['reg_cl']
    transport = ot.da.SinkhornLpl1Transport(reg_e=param_ot['reg_e'], reg_cl=param_ot['reg_cl'], norm="median")
    transport.fit(Xs=X_source, ys=y_source, Xt=X_target)
    if not target_to_source:
        transp_Xs = transport.transform(Xs=X_source)
        return transp_Xs
    else:
        transp_Xt = transport.inverse_transform(Xt=X_target)
        return transp_Xt


def print_pickle(filename, type=""):
    if type == "results" :
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
    else :
        print("Data saved in", filename)
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        print(data)



def main(argv, adaptation=False, filename=""):
    listParams = {
        "XGBoost": listP(
            {'max_depth': range(1, 6),
             # 'eta': [10**(-i) for i in range(1, 5)],
             # 'subsample': np.arange(0.1, 1, 0.1),
             # 'colsample_bytree': np.arange(0.1, 1, 0.1),
             # 'gamma': range(0, 21),
             'num_boost_round': range(100, 1001, 100)
             })
    }

    nbFoldValid = 5
    seed = 1
    if len(argv) == 2:
        seed = int(argv[1])

    results = {}
    for dataset in ['abalone8']:  # ['abalone20', 'abalone17', 'satimage', 'abalone8']:  #
        X, y = data_recovery(dataset)
        pctPos = 100 * len(y[y == 1]) / len(y)
        dataset = "{:05.2f}%".format(pctPos) + " " + dataset
        print(dataset)
        np.random.seed(seed)
        random.seed(seed)

        # Split the dataset between the source and the target(s)
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

        if adaptation:
            pass
            # Xsource = ot_adaptation(Xsource, ysource, Xtarget)
            # Xtarget = ot_adaptation(Xsource, ysource, Xtarget, True)

        # From the source, training and test set are created
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xsource, ysource,
                                                        shuffle=True,
                                                        stratify=ysource,
                                                        test_size=0.3)

        # For cross validation
        skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        foldsTrainValid = list(skf.split(Xtrain, ytrain))
        results[dataset] = {}

        for algo in listParams.keys():

            start = time.time()
            '''if len(listParams[algo]) > 1:  # Cross validation
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
                param = listParams[algo][0]'''

            # BEGIN EXPE

            param = listParams[algo][0]
            cross_val_result = transfer_cross_validation_trg_to_src(Xsource, ysource, Xtarget, param, "exp_1.pklz")
            param_ot = {'reg_e': cross_val_result['reg_e'], 'reg_cl': cross_val_result['reg_cl']}
            Xtarget = ot_adaptation(Xsource, ysource, Xtarget, param_ot, True)

            # END EXPE
            apTrain, apTest, apClean, apTarget = applyAlgo(algo, param,
                                                           Xtrain, ytrain,
                                                           Xtest, ytest,
                                                           Xtarget, ytarget,
                                                           Xclean)
            results[dataset][algo] = (apTrain, apTest, apClean, apTarget, param, param_ot)
            print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
                  "Test AP {:5.2f}".format(apTest),
                  "Clean AP {:5.2f}".format(apClean),
                  "Target AP {:5.2f}".format(apTarget), param, param_ot,
                  "in {:6.2f}s".format(time.time() - start))
        if not os.path.exists("results"):
            try:
                os.makedirs("results")
            except:
                pass
        if filename == "":
            filename = f"./results/res{seed}.pklz"
        else:
            filename = f"./results/" + filename
        f = gzip.open(filename, "wb")
        pickle.dump(results, f)
        f.close()


if __name__ == '__main__':
    # configure debugging tool
    ic.configureOutput(includeContext=True)

    main(sys.argv, adaptation=True, filename="res_cross_val_ot.pklz")
    # main(sys.argv, filename=res.pklz")

    # print_pickle("results/res_cross_val_ot.pklz", "results")
    # print_pickle("OT_cross_valid_results/exp_1.pklz", "OT_cross_valid_results")

    ''' 
    print_pickle("results/res2.pklz")
    print_pickle("results/res2_transport.pklz")
    print_pickle("results/res2_transportEMD.pklz")
    '''
