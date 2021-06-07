from datetime import datetime

from main import adapt_domain, train_model
from utils import *
from optimal_transport import ot_cross_validation


def expe_multi_source_optimal_transport(dataset, source_path, target_path, hyperparameter_file, filename, algo,
                                        adaptation_method, cv_with_true_labels, transpose, nb_iteration_cv,
                                        select_feature, cluster, rescale_type="", rescale=False):
    normalizer = None

    X_source_1, y_source_1, X_source_2, y_source_2, X_source_3, \
    y_source_3, index1, index2, index3 = import_source_per_year(source_path, select_feature)

    if rescale_type != "":
        X_source_1 = get_normalizer_data(X_source_1, rescale_type)
        X_source_2 = get_normalizer_data(X_source_2, rescale_type)
        X_source_3 = get_normalizer_data(X_source_3, rescale_type)

    list_X_source = [X_source_1, X_source_2, X_source_3]
    list_y_source = [y_source_1, y_source_2, y_source_3]

    _, y_source, _ = import_dataset(source_path, select_feature)
    X_target, y_target, _ = import_dataset(target_path, select_feature)
    if rescale_type != "":
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
    possible_reg_e = [0.1, 1, 5, 10, 12, 15]
    possible_reg_cl = [0.05, 0.1, 0.5, 1, 5, 8, 10]
    param_to_cross_valid = {'reg_e': possible_reg_e, 'reg_cl': possible_reg_cl}
    cv_filename = f"./" + repo_name + "/cross_validation_" + file_id

    list_trans_X_source = []
    param_transport = {}
    param_transport_true_label = {}
    for i in range(3):
        param_transport, param_transport_true_label = ot_cross_validation(list_X_source[i], list_y_source[i], X_target,
                                                                          hyperparameter_file,
                                                                          param_to_cross_valid,
                                                                          normalizer,
                                                                          ot_type=adaptation_method,
                                                                          rescale=rescale,
                                                                          transpose_plan=False,
                                                                          filename="", y_target=y_target,
                                                                          cv_with_true_labels=cv_with_true_labels,
                                                                          nb_training_iteration=nb_iteration_cv,
                                                                          cluster=cluster)

        X_source, X_target, X_clean = adapt_domain(list_X_source[i], list_y_source[i], X_target, X_clean,
                                                   param_transport, transpose, adaptation_method)
        list_trans_X_source.append(X_source)

    # Creation of the filename
    if filename == "":
        if not transpose:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_classic_" + rescale_type + "_" + adaptation_method + "_" + algo + file_id
        else:
            filename = f"./" + repo_name + "/" + dataset + \
                       "_" + adaptation_method + "_" + algo + file_id

    temp_trans_X_source = np.append(list_trans_X_source[0], list_trans_X_source[1], axis=0)
    temp_trans_X_source = np.append(temp_trans_X_source, list_trans_X_source[2], axis=0)
    X_source = temp_trans_X_source

    params_model = cross_validation_model(X_source, y_source, hyperparameter_file, nbFoldValid=3)

    apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_clean, params_model,
                                                     normalizer, False, algo)
    save_results(adaptation_method, dataset, algo, apTrain, apTest, apClean, apTarget, params_model,
                 param_transport, start, filename, results, param_transport_true_label)


if __name__ == "__main__":
    i = 1
    model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(i) + "_fraude2_best_model_and_params.csv"
    source = "./datasets_fraude2/source_" + str(i) + "_fraude2.csv"
    target = "./datasets_fraude2/target_" + str(i) + "_fraude2.csv"

    filename = f"./results2805/multisource_experiment_with_3_transport_plan"

    name = "cluster" + str(i) + "_fraude2"
    expe_multi_source_optimal_transport(name, source, target, model_hyperparams, filename, "XGBoost",
                                        "OT", cv_with_true_labels=True, transpose=False, nb_iteration_cv=2,
                                        select_feature=False, cluster=1, rescale_type="Min_Max", rescale=True)