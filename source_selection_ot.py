import datetime
import os
import time
from sys import argv

from main import import_source_per_year, import_dataset, import_hyperparameters, adapt_domain, cross_validation_model, \
    train_model, save_results
import numpy as np


"""
for cluster in 12
do
    for combination in 1 2 3 12 13 23
    do
        python source_selection_ot.py $cluster $combination &
    done
done
"""

def launch_expe_partial_source(flag, cluster=12):
    source_path = "./datasets_fraude2/source_" + str(cluster) + "_fraude2.csv"
    target_path = "./datasets_fraude2/target_" + str(cluster) + "_fraude2.csv"
    

    X_1, y_1, X_2, y_2, X_3, y_3, _, _, _ = import_source_per_year(source_path, False)
    X_target, y_target, _ = import_dataset(target_path, False)

    if flag == "1":
        X_source = X_1
        y_source = y_1
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)
    elif flag == "2":
        X_source = X_2
        y_source = y_2
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)
    elif flag == "3":
        X_source = X_3
        y_source = y_3
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)
    elif flag == "12":
        X_source = X_1
        y_source = y_1
        y_source = np.append(y_source, y_2)
        X_source = np.vstack((X_source, X_2))
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)
    elif flag == "13":
        X_source = X_1
        y_source = y_1
        X_source = np.vstack((X_source, X_3))
        y_source = np.append(y_source, y_3)
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)
    elif flag == "23":
        X_source = X_2
        y_source = y_2
        X_source = np.vstack((X_source, X_3))
        y_source = np.append(y_source, y_3)
        source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag)

def source_selected_ot(X_source, y_source, X_target, y_target, cluster, flag):
    model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(cluster) + "_fraude2_best_model_and_params.csv"
    X_clean = X_target
    params_model = import_hyperparameters("XGBoost", model_hyperparams)
    results = {}
    start = time.time()

    repo_name = "expe_selecting_source"
    if not os.path.exists(repo_name):
        try:
            os.makedirs(repo_name)
        except:
            pass
    dataset = "cluster12_fraude2"
    results[dataset] = {}
    param_transport_true_label = {}
    param_transport = {}  # for the pickle

    filename = f"./" + repo_name + "/" + dataset + "expe_OT_src" + flag

    param_transport, param_transport_true_label = {'reg_e': 10, 'reg_cl': 0.1}, None
    X_source, X_target, X_clean = adapt_domain(X_source, y_source, X_target, X_clean, param_transport, False, "OT")

    apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, params_model,
                                                     None, False, "XGBoost")

    results = save_results("OT", dataset, "XGBoost", apTrain, apTest, apClean, apTarget, params_model,
                           param_transport, start, filename, results, param_transport_true_label)


if __name__ == '__main__':
    launch_expe_partial_source(argv[2], int(argv[1]))
