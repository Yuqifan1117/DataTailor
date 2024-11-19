from time import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import torch
import json
import math
from datetime import datetime
import csv
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def extract_upper_triangle(distance_matrix_np):
    n = distance_matrix_np.shape[0]
    pdist_array = np.empty((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pdist_array[idx] = distance_matrix_np[i, j]
            idx += 1
    return pdist_array

def get_posture(x, y, n):
    return x * n - x * (x + 1) // 2 + y - x - 1

def get_distance(distance_matrix_np):
    n = distance_matrix_np.shape[0]
    pdist_array = np.empty((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pdist_array[idx] = distance_matrix_np[i, j]
            idx += 1
    return pdist_array

def output(avg_value, save_name, json_info, num, block_number, belong):
    print(save_name)
    
    sorted_indices = sorted(avg_value, key=lambda tup: tup[1], reverse=True)
    
    selected_indices = []
    final_num = int(num * 0.075)
    for i in range(0, final_num):
        selected_indices.append(sorted_indices[i][0])

    sum_block = torch.tensor([0.] * block_number)
    cluster_block = torch.tensor([0.] * block_number)
    for i in range(0, final_num):
        sum_block[belong[sorted_indices[i][0]]] += 1;
    for i in range(0, num):
        cluster_block[belong[sorted_indices[i][0]]] += 1
        
    for i in range(0, block_number):
        print(i, sum_block[i] / cluster_block[i])
        
    selected_indices = sorted(selected_indices)

    save_info=[json_info[i] for i in selected_indices]
    with open(save_name + ".json", 'w') as fp:
        json.dump(save_info, fp, indent=4)

def work(X, Y):

    linear_coeffs = np.polyfit(X, Y, 1)
    linear_fit = np.polyval(linear_coeffs, X)

    quad_coeffs = np.polyfit(X, Y, 2)
    quad_fit = np.polyval(quad_coeffs, X)
    
    def log_func(X, a, b):
        return a * np.log(b * X)

    log_coeffs, _ = curve_fit(log_func, X, Y)
    log_fit = log_func(X, *log_coeffs)

    def calc_std(y_true, y_pred):
        return np.std(y_true - y_pred)

    linear_std = calc_std(Y, linear_fit)
    quad_std = calc_std(Y, quad_fit)
    log_std = calc_std(Y, log_fit)

    print(f"Linear fit: Y = {linear_coeffs[0]:.5f}X + {linear_coeffs[1]:.5f}, Std: {linear_std:.5f}")
    print(f"Quadratic fit: Y = {quad_coeffs[0]:.5f}X^2 + {quad_coeffs[1]:.5f}X + {quad_coeffs[2]:.5f}, Std: {quad_std:.5f}")
    print(f"Logarithmic fit: Y = {log_coeffs[0]:.5f} * log({log_coeffs[1]:.5f}X), Std: {log_std:.5f}")

    plt.scatter(X, Y, label='Data')
    plt.plot(X, linear_fit, label=f'Linear Fit (Std: {linear_std:.2f})', color='red')
    plt.plot(X, quad_fit, label=f'Quadratic Fit (Std: {quad_std:.2f})', color='blue')
    plt.plot(X, log_fit, label=f'Logarithmic Fit (Std: {log_std:.2f})', color='purple')
    plt.legend()

def get_relationship_value(load_name_weights, load_name_clustering):
    num = 665298
    print(datetime.now())
    mix_st = [0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610]
    mix_ed = [57668, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297]
    final_num_cho = []

    json_root=r'llava_v1_5_mix665k.json'
    with open(json_root, 'r') as f:
        json_info = json.load(f)
    save_DATATAILOR=json_info[mix_st[6]:(mix_ed[6]+1)]

    with open("llava_v1_5_mix665k_refcoco.json", 'w') as fp:
        json.dump(save_DATATAILOR, fp, indent=4)
    exit(0)
    belong = []
    sum = 0.
    l1 = torch.load("l1.pt")    
    s1 = torch.load("s1.pt")
    t1 = torch.load("t1.pt")

    for i in range(0, 12):
        mm = torch.mean(t1[mix_st[i]:(mix_ed[i] + 1)]) * torch.mean(t1[mix_st[i]:(mix_ed[i] + 1)]) 
        chose = (mix_ed[i] - mix_st[i] + 1) * mm
        sum += chose
    one = 0.075 * num / sum
    for i in range(0, 12):
        mm = torch.mean(t1[mix_st[i]:(mix_ed[i] + 1)]) * torch.mean(t1[mix_st[i]:(mix_ed[i] + 1)]) 
        chose = (mix_ed[i] - mix_st[i] + 1) * mm
        select = one * chose
        final_num_cho.append(select)
        print(select)
    weight1 = []
    weight_part = []
    weight_block = []
    length = []
    
    s_d = torch.tensor([0.] * num)
    s_n = torch.tensor([0.] * num)
    s_n1 = torch.tensor([0.] * num)
    s = torch.tensor([1.] * num)

    labels_pred = torch.cat(torch.load("label_1.0_dis.pt"))

    block_number = torch.max(labels_pred)
    
    labels_pred -= 1
    _res = []
    print("Start init!!!")
    for i in range(0, block_number):
        __res = (labels_pred == i).nonzero(as_tuple=True)[0].tolist()
        _res.append(__res)
    
    
    v = s1
    r = torch.load("temporary.pt")
    sd1 = torch.load("010sd1.pt")
    sd2 = torch.load("010sd2.pt")
    sd3 = torch.load("010sd3.pt")
    sd4 = torch.load("010sd4.pt")
    sn_si = torch.load("010sn_si.pt")
    sn_sj = torch.load("010sn_sj.pt")
    sn = torch.load("sn.pt")
    
    R = np.array(torch.load("total_round.pt")) / 2
    
    print("Start calculate final") 
    
    selected_DATATAILOR = []
      
    s = torch.load("nom_s.pt").numpy()  
    R = np.array(torch.load("total_round.pt")) / 2
    ans1 = torch.tensor([0.] * num)
    ans2 = torch.tensor([0.] * num)
    ans3 = torch.tensor([0.] * num)
    for data_type in range(0, 12):
        pair_DATATAILOR = []
        total = int(final_num_cho[data_type] + 0.5)
        print(total)
        _max_v = - 1e18
        _min_v = 1e18
        _max_r = - 1e18
        _min_r = 1e18
        _sum_normalized_v = 0.
        _sum_normalized_r = 0.
        
        i = data_type
        s1_data = s1[mix_st[i]:(mix_ed[i]+1)]
        sn_si_data = sn_si[mix_st[i]:(mix_ed[i]+1)]
        sn_sj_data = sn_sj[mix_st[i]:(mix_ed[i]+1)]
        sd1_data = sd1[mix_st[i]:(mix_ed[i]+1)]
        sd3_data = sd3[mix_st[i]:(mix_ed[i]+1)]
        s1_data = np.argsort(np.argsort(s1_data)) / (mix_ed[i] + 1 - mix_st[i])
        sn_si_data = np.argsort(np.argsort(sn_si_data)) / (mix_ed[i] + 1 - mix_st[i])
        sn_sj_data = np.argsort(np.argsort(sn_sj_data)) / (mix_ed[i] + 1 - mix_st[i])
        sd1_data = np.argsort(np.argsort(sd1_data)) / (mix_ed[i] + 1 - mix_st[i])
        sd3_data = np.argsort(np.argsort(sd3_data)) / (mix_ed[i] + 1 - mix_st[i])
        R_data = R[mix_st[i]:(mix_ed[i]+1)]
        for pos_i in range(0, len(_res[data_type])):
            i = _res[data_type][pos_i]
            pair_DATATAILOR.append([(s1_data[pos_i] * R[i]
                                     +sn_sj_data[pos_i]
                                     +sd1_data[pos_i])
                                  /  (R[i] + 2)
                                  ,  i])        
            ans1[i] = s1_data[pos_i]
            ans2[i] = sn_sj_data[pos_i] 
            ans3[i] = sd1_data[pos_i] 

        pair_DATATAILOR = sorted(pair_DATATAILOR, key=lambda tup: tup[0], reverse=True)
        for i in range(0, total):
            selected_DATATAILOR.append(pair_DATATAILOR[i][1])
            
    selected_DATATAILOR = sorted(selected_DATATAILOR)
    print(len(selected_DATATAILOR))

    json_root=r'llava_v1_5_mix665k.json'
    with open(json_root, 'r') as f:
        json_info = json.load(f)
    save_DATATAILOR=[json_info[i] for i in selected_DATATAILOR]

    with open("llava_v1_5_mix665k_0.075-ablation5.json", 'w') as fp:
        json.dump(save_DATATAILOR, fp, indent=4)
        
    exit(0)

if __name__ == '__main__':
    load_name_weights = "total_global_llava665k_hidden_state.pt"
    load_name_clustering = "total_clustering_global_llava665k_hidden_state_split_1000.pt"
    v = get_instance_value()
    print("Start data selection!!!")
    get_relationship_value(load_name_weights, load_name_clustering)
