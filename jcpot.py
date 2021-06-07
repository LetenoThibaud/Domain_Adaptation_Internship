from sklearn.preprocessing import Normalizer
from main import import_source_per_year, import_dataset, import_hyperparameters, cross_validation_model, train_model
import numpy as np
from scipy.spatial.distance import cdist

cluster = 1
model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(cluster) + "_fraude2_best_model_and_params.csv"
source = "./datasets_fraude2/source_" + str(cluster) + "_fraude2.csv"
target = "./datasets_fraude2/target_" + str(cluster) + "_fraude2.csv"
model_hyperparams = "~/restitution/9_travaux/dm/2020/modeles_seg/modeles_seg_new/cluster" + str(cluster) + "_fraude2_best_model_and_params.csv"

def projR(gamma,p):
    #return np.dot(np.diag(p/np.maximum(np.sum(gamma,axis=1),1e-10)),gamma)
    return np.multiply(gamma.T,p/np.maximum(np.sum(gamma,axis=1),1e-10)).T

def projC(gamma,q):
    #return (np.dot(np.diag(q/np.maximum(np.sum(gamma,axis=0),1e-10)),gamma.T)).T
    return np.multiply(gamma,q/np.maximum(np.sum(gamma,axis=0),1e-10))

def estimateTransport(all_Xr,Xt,reg,numItermax = 100, tol_error=1e-7):
    nbdomains = len(all_Xr)
    # we then build, for each source domain, specific information
    all_domains = {}
    for d in range(nbdomains):
        all_domains[d] = {}
        # get number of elements for this domain
        nb_elem = all_Xr[d].shape[0]
        all_domains[d]['nbelem'] = nb_elem
        # build the distance matrix
        M = cdist(all_Xr[d],Xt,metric='sqeuclidean')
        M = M/np.median(M)
        K = np.exp(-M/reg)
        all_domains[d]['K'] = K
        all_domains[d]['w'] = np.ones(nb_elem).astype(float)/nb_elem

    distrib = np.ones(Xt.shape[0])/Xt.shape[0]

    cpt=0
    log = {}

    while (cpt<numItermax):
        for d in range(nbdomains):
            all_domains[d]['K'] = projC(all_domains[d]['K'],distrib)
            all_domains[d]['K'] = projR(all_domains[d]['K'],all_domains[d]['w'])
        cpt=cpt+1

    log['all_domains']=all_domains
    return log


def estimateTranspPoints(Xt,log):
    nd =len(log['all_domains'])
    all_Xr_transp=[]
    for d in range(nd):
        transp = log['all_domains'][d]['K']
        transp1 = np.dot(np.diag(1/(np.sum(transp,1)+1e-8)),transp)
        all_Xr_transp.append(np.array(np.dot(transp1,Xt)))
    return all_Xr_transp


def cheat_cv(X_sources, y_source, X_target, y_target, possible_reg_e):
    for reg_e in possible_reg_e:
        log = estimateTransport(X_sources, X_target, reg_e)
        transp_pts = estimateTranspPoints(X_target, log)

        print(transp_pts)






# import data
X_source_1, y_source_1, X_source_2, y_source_2, X_source_3, y_source_3, index1, index2, index3 = import_source_per_year(
        source, False)
X_target, y_target, _ = import_dataset(target, False)
X_source, y_source, _ = import_dataset(source, False)

# normalize data
X_source_1 = Normalizer().fit(X_source_1).transform(X_source_1)
X_source_2 = Normalizer().fit(X_source_2).transform(X_source_2)
X_source_3 = Normalizer().fit(X_source_3).transform(X_source_3)
X_target = Normalizer().fit(X_target).transform(X_target)

# group sources in list to be transported
list_X_source = [X_source_1, X_source_2, X_source_3]
list_y_source = [y_source_1, y_source_2, y_source_3]

params_model = import_hyperparameters("XGBoost", model_hyperparams)

possible_reg_e = [0.1, 1, 5, 10]

log = estimateTransport(list_X_source, X_target, possible_reg_e[1])
transp_pts = estimateTranspPoints(X_target, log)

temp_trans_X_source = np.append(transp_pts[0], transp_pts[1], axis=0)
temp_trans_X_source = np.append(temp_trans_X_source, transp_pts[2], axis=0)
X_source = temp_trans_X_source

# params_model = cross_validation_model(X_source, y_source, model_hyperparams)

rescale = False
normalizer = None
apTrain, apTest, apClean, apTarget = train_model(X_source, y_source, X_target, y_target, X_target, params_model,
                                                normalizer, rescale, "XGBoost")

print("AP train : ", apTrain)
print("AP test : ", apTest)
print("AP target : ", apTest)
