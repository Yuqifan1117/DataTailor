import numpy as np
import torch
import math
from scipy.cluster.hierarchy import fcluster, inconsistent
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from scipy.cluster import _hierarchy

def extract_upper_triangle(distance_matrix_np):
    n = distance_matrix_np.shape[0]
    pdist_array = np.empty((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pdist_array[idx] = distance_matrix_np[i, j]
            idx += 1
    return pdist_array
    
def linkage(pdist_np, y, method='single', metric='euclidean', optimal_ordering=False):
    n = len(y)
    result = _hierarchy.nn_chain(pdist_np, n, 5)
    return result
        
def make_split_665k_clustering():
    param = 0.1
    st = [0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610]
    ed = [57668, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297]
    cnt = len(st)
    cur_block_number = 0
    labels_pred = []
    weights = torch.load("total_global_llava665k_hidden_state.pt")
    print("Start make clustering")
    for part in range(0, cnt):
        weight = []
        _weight = []
        num = ed[part] - st[part] + 1
        for i in range(st[part], ed[part] + 1):
            weight.append(weights[i].double().tolist())
            _weight.append(weights[i].double())
        weight = torch.tensor(weight, dtype = torch.float64)
        pdist_np_ = np.empty((num * (num - 1)) // 2)
        dist = torch.cdist(weight, weight, p = 2)
        dist_np = dist.cpu().numpy()
        pdist_np = extract_upper_triangle(dist_np)
        Z = linkage(pdist_np, _weight, method='ward')
        distance_threshold = Z[len(Z) - 1][2]
        labels_pred_part = torch.tensor(fcluster(Z, t=distance_threshold * param, criterion='distance')) + cur_block_number
        cur_block_number = torch.max(labels_pred_part).item()
        labels_pred.append(labels_pred_part)

    torch.save(labels_pred, "labels_pred.pt")

if __name__ == '__main__':
    make_split_665k_clustering()
