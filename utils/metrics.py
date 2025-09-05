import numpy as np 
from bayesian_torch.utils.util import predictive_entropy
import torch
import sklearn
import sklearn.metrics
import uncertainty_metrics.numpy as um
from torch_uncertainty.metrics.classification import FPR95

def AULC(accs, uncertainties):
    # copied from: https://github.com/cvlab-epfl/zigzag/blob/main/exps/notebooks/mnist_classification.ipynb
    idxs = np.argsort(uncertainties)
    uncs_s = uncertainties[idxs]
    error_s = accs[idxs]

    mean_error = error_s.mean()
    error_csum = np.cumsum(error_s)

    Fs = error_csum / np.arange(1, len(error_s) + 1)
    s = 1 / len(Fs)
    return -1 + s * Fs.sum() / mean_error, Fs

def rAULC(uncertainties, accs):
    perf_aulc, Fsp = AULC(accs, -accs.astype("float"))
    curr_aulc, Fsc = AULC(accs, uncertainties)
    return curr_aulc / perf_aulc, Fsp, Fsc

def compute_AUCs(uc_id, uc_labels,uc_ood):
    id_labels = np.zeros_like(uc_id)
    ood_labels = np.ones_like(uc_ood)
    uc_labels_ood = np.concatenate([id_labels, ood_labels])
    uc_values = np.concatenate([uc_id, uc_ood])
    
    raulc, r1, r2 = rAULC(np.array(uc_id), np.array(uc_labels))
    roc_auc = sklearn.metrics.roc_auc_score(np.array(uc_labels_ood), np.array(uc_values))
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(np.array(uc_labels_ood), np.array(uc_values))
    pr_auc = sklearn.metrics.auc(recall, precision)
    return raulc, roc_auc, pr_auc

def compute_FPR95(uc_id, uc_ood):
    fpr95 = FPR95(pos_label=1)
    fpr95.update(uc_id, torch.zeros_like(uc_id))
    fpr95.update(uc_ood, torch.ones_like(uc_ood))
    result =  fpr95.compute().item()
    return result
