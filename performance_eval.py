import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from scipy.stats import pearsonr


def visualise_L(L_orig, L_opt):
#     L_orig = -(L_orig - np.diag(np.diag(L_orig)))
#     L_opt = -(L_opt - np.diag(np.diag(L_opt)))
    
    ax1 = plt.subplot(121)
    ax1.imshow(L_orig)
    ax1.title.set_text(r"True $L$")
    ax2 = plt.subplot(122)
    ax2.imshow(L_opt)
    ax2.title.set_text(r"Predicted $L$")
    plt.show()
    
def graph_learning_perf_eval(L_orig, L_pred):
    """
    L_orig : groundtruth graph Laplacian
    L_pred : learned graph Laplacian
    """
    # evaluate the performance of graph learning algorithms
    
    n = L_orig.shape[0]
    idx_non_diag = np.triu_indices(n, 1) # excluding diagonal
    
    L_orig_nd = np.diag(np.diag(L_orig)) - L_orig
    edges_groundtruth = (L_orig_nd > 1e-4)[idx_non_diag] + 0

    L_pred_nd = np.diag(np.diag(L_pred)) - L_pred
    edges_learned = (L_pred_nd > 1e-4)[idx_non_diag] + 0
        
    condition_positive = np.sum(edges_groundtruth)
    prediction_positive = np.sum(edges_learned)
    true_positive = np.sum(np.logical_and(edges_groundtruth, edges_learned))
    print(f"condition positive:{condition_positive}, prediction positive:{prediction_positive}, true_positive:{true_positive}")
    
    precision = true_positive / prediction_positive
    recall = true_positive / condition_positive
    
    if precision == 0 or recall == 0:
        f = 0
    else:
        f = 2 * precision * recall / (precision + recall)
    
    NMI = nmi(edges_groundtruth, edges_learned)
    
    R, _ = pearsonr(L_orig[idx_non_diag], L_pred[idx_non_diag])

    return precision, recall, f, NMI, R