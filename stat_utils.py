from scipy.stats import ks_2samp
from main import import_source_per_year, import_dataset
import numpy as np
import pandas as pd

def ks_test(distrib_1, distrib_2):
    return ks_2samp(distrib_1, distrib_2)

def compare_distribution_ks(source_path, target_path):
    X_1, _, X_2, _, X_3, _, _, _, _ = import_source_per_year(source_path, False)
    X_target, _, _ = import_dataset(target_path, False)

    results = pd.DataFrame([None]*X_1.shape[1])
    for distrib_1 in [X_1, X_2, X_3, X_target] :
        result = []
        for distrib_2 in [X_1, X_2, X_3, X_target] :
            for col in range(distrib_1.shape[1]):
                ks, pval = ks_test(distrib_1[:, col], distrib_2[:, col])
                result.append(pval)
                print(pval)
        result.append(np.mean(result)) 
        print(result)
        results.append(result)
    results.to_csv("ks_test_cluster12")


if __name__ == '__main__':
    cluster = 12
    source_path = "./datasets_fraude2/source_" + str(cluster) + "_fraude2.csv"
    target_path = "./datasets_fraude2/target_" + str(cluster) + "_fraude2.csv"

    compare_distribution_ks(source_path, target_path)