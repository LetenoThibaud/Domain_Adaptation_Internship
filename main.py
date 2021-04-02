import os
import csv
import gzip
import pickle
import random
import sys
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


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
        X = data[:, np.arange(d-1)].astype(float)
        y = data[:, d-1]
        y = y.astype(int)
        y[y != 4] = 0
        y[y == 4] = 1
    return X, y


def listP(dic):  # Create grid of parameters given parameters ranges
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
    hess = np.ones(len(preds))*0.1
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
    return (average_precision_score(ytrain, rankTrain)*100,
            average_precision_score(ytest, rankTest)*100,
            average_precision_score(ytarget, rankClean)*100,
            average_precision_score(ytarget, rankTarget)*100)


listParams = {
              "XGBoost": listP(
                    {'max_depth': range(1, 6),
                     #'eta': [10**(-i) for i in range(1, 5)],
                     #'subsample': np.arange(0.1, 1, 0.1),
                     #'colsample_bytree': np.arange(0.1, 1, 0.1),
                     #'gamma': range(0, 21),
                     'num_boost_round': range(100, 1001, 100)
                     })
              }

nbFoldValid = 5
seed = 1
if len(sys.argv) == 2:
    seed = int(sys.argv[1])

results = {}
for dataset in ['abalone20', 'abalone17', 'satimage', 'abalone8']:
    X, y = data_recovery(dataset)
    pctPos = 100*len(y[y == 1])/len(y)
    dataset = "{:05.2f}%".format(pctPos) + " " + dataset
    print(dataset)
    np.random.seed(seed)
    random.seed(seed)

    Xsource, Xtarget, ysource, ytarget = train_test_split(X, y, shuffle=True,
                                                          stratify=y,
                                                          test_size=0.51)
    Xclean = Xtarget.copy()
    for feat, coef in [(2, 0.1), (3, 10), (4, 0)]:
        if coef != 0:
            Xtarget[:, feat] = Xtarget[:, feat]*coef
        else:
            Xtarget[np.random.choice(len(Xtarget), int(len(Xtarget)/2)),
                    feat] = 0
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xsource, ysource,
                                                    shuffle=True,
                                                    stratify=ysource,
                                                    test_size=0.3)
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
        apTrain, apTest, apClean, apTarget = applyAlgo(algo, param,
                                                       Xtrain, ytrain,
                                                       Xtest, ytest,
                                                       Xtarget, ytarget,
                                                       Xclean)
        results[dataset][algo] = (apTrain, apTest, apClean, apTarget)
        print(dataset, algo, "Train AP {:5.2f}".format(apTrain),
              "Test AP {:5.2f}".format(apTest),
              "Clean AP {:5.2f}".format(apClean),              
              "Target AP {:5.2f}".format(apTarget), param,
              "in {:6.2f}s".format(time.time()-start))
    if not os.path.exists("results"):
        try:
            os.makedirs("results")
        except:
            pass
    f = gzip.open(f"./results/res{seed}.pklz", "wb")
    pickle.dump(results, f)
    f.close()
