from time import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, inconsistent
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import torch
import json
import math
from datetime import datetime

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

def make_clustering(load_name, save_name):
    weights = torch.load(load_name)
    block_number = 30

    weight1 = []
    weight2 = []
    num = weights.shape[0]
    for i in range(0, num):
        weight1.append(weights[i].double().tolist())
        weight2.append(weights[i].double())

    weight = torch.tensor(weight1, dtype=torch.float64)

    pdist_np_ = np.empty((num * (num - 1)) // 2)        
    dist = torch.cdist(weight, weight, p = 2)
    dist_np = dist.cpu().numpy()
    pdist_np = extract_upper_triangle(dist_np)
    Z = linkage(pdist_np, weight2, method='ward')
    labels_pred = fcluster(Z, t = block_number, criterion = 'maxclust')
    labels_pred = torch.tensor(labels_pred)
    print(labels_pred)
    print(len(labels_pred))
    torch.save(labels_pred, save_name)
    
def make_split_clustering(load_name, save_name):
    weights = torch.load(load_name)
    st = [0, 57669, 80909]
    ed = [57668, 80908, 157711]
    blk_number = [11, 4, 15]
    labels_pred = []
    cur_block_number = 0
    cnt = 3
    for part in range(0, cnt):
        weight1 = []
        weight2 = []
        num = ed[part] - st[part] + 1
        block_number = blk_number[part]
        for i in range(st[part], ed[part] + 1):
            weight1.append(weights[i].double().tolist())
            weight2.append(weights[i].double())
        
        weight = torch.tensor(weight1, dtype=torch.float64)
        
        pdist_np_ = np.empty((num * (num - 1)) // 2)        
        dist = torch.cdist(weight, weight, p = 2)
        dist_np = dist.cpu().numpy()
        pdist_np = extract_upper_triangle(dist_np)
        Z = linkage(pdist_np, weight2, method='ward')
        labels_pred_part = fcluster(Z, t = block_number, criterion = 'maxclust')
        labels_pred_part = torch.tensor(labels_pred_part) + cur_block_number
        print(labels_pred_part)
        if part == 0:
            labels_pred = labels_pred_part.clone().detach()
        else:
            labels_pred = torch.cat((labels_pred, labels_pred_part), dim = 0)
        cur_block_number += block_number
    print(labels_pred)
    print(len(labels_pred))
    torch.save(labels_pred, save_name)

def make_split_665k_clustering(load_name, save_name):
    import math
    st = [0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610]
    ed = [62000, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297] #57668
    cnt = len(st)
    blk_number_005 = []
    blk_number_010 = []
    blk_number_025 = []
    blk_number_050 = []
    blk_number_075 = []
    blk_number_100 = []
    
    cur_block_number_005 = 0
    cur_block_number_010 = 0
    cur_block_number_025 = 0
    cur_block_number_050 = 0
    cur_block_number_075 = 0
    cur_block_number_100 = 0

    labels_pred_005 = []
    labels_pred_010 = []
    labels_pred_025 = []
    labels_pred_050 = []
    labels_pred_075 = []
    labels_pred_100 = []
    
    weights = torch.load("total_global_llava665k_hidden_state.pt")

    for part in range(0, cnt):
        weight1 = []
        weight2 = []
        num = ed[part] - st[part] + 1
        for i in range(st[part], ed[part] + 1):
            weight1.append(weights[i].double().tolist())
            weight2.append(weights[i].double())
        
        weight = torch.tensor(weight1, dtype=torch.float64)
        
        pdist_np_ = np.empty((num * (num - 1)) // 2)        

        dist = torch.cdist(weight, weight, p = 2)
        
#        norms = torch.norm(weight, dim=1, keepdim=True)
#        dot_product = torch.mm(weight, weight.t())
#        dist = dot_product / (norms * norms.t())
#        dist = (1 - dist) / 2

        dist_np = dist.cpu().numpy()
        pdist_np = extract_upper_triangle(dist_np)
        Z = linkage(pdist_np, weight2, method='ward')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance'))
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())
            
        Z = linkage(pdist_np, weight2, method='single')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance')) 
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())

        Z = linkage(pdist_np, weight2, method='complete')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance')) 
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())

        Z = linkage(pdist_np, weight2, method='average')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance')) 
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())

        Z = linkage(pdist_np, weight2, method='weighted')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance')) 
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())

        Z = linkage(pdist_np, weight2, method='median')
        print(Z[len(Z) - 1])
        distance_threshold = Z[len(Z) - 1][2]
        print(distance_threshold)  
        labels_pred_part_050 = torch.tensor(fcluster(Z, t=distance_threshold * 0.50, criterion='distance')) 
        for w in range(0, 10):
            print((labels_pred_part_050 == w).sum().item())

#        labels_pred_part_005 = torch.tensor(fcluster(Z, t=distance_threshold * 0.05, criterion='distance')) + cur_block_number_005
#        cur_block_number_005 = torch.max(labels_pred_part_005).item()
#        blk_number_005.append(cur_block_number_005)

#        labels_pred_part_010 = torch.tensor(fcluster(Z, t=distance_threshold * 0.10, criterion='distance')) + cur_block_number_010
#        cur_block_number_010 = torch.max(labels_pred_part_010).item()
#        blk_number_010.append(cur_block_number_010)

#        labels_pred_part_025 = torch.tensor(fcluster(Z, t=distance_threshold * 0.25, criterion='distance')) + cur_block_number_025
#        cur_block_number_025 = torch.max(labels_pred_part_025).item()
#        blk_number_025.append(cur_block_number_025)


#        labels_pred_part_075 = torch.tensor(fcluster(Z, t=distance_threshold * 0.75, criterion='distance')) + cur_block_number_075
#        cur_block_number_075 = torch.max(labels_pred_part_075).item()
#        blk_number_075.append(cur_block_number_075)

#        labels_pred_part_100 = torch.tensor(fcluster(Z, t=distance_threshold * 1.0, criterion='distance')) + cur_block_number_100
#        cur_block_number_100 = torch.max(labels_pred_part_100).item()
#        blk_number_100.append(cur_block_number_100)


        torch.save(labels_pred_part_005, "check0050.pt")
        torch.save(labels_pred_part_010, "check0100.pt")
        torch.save(labels_pred_part_025, "check0250.pt")
        torch.save(labels_pred_part_050, "check0500.pt")
        torch.save(labels_pred_part_075, "check0750.pt")
        torch.save(labels_pred_part_100, "check1000.pt")
        exit(0)

        labels_pred_005.append(labels_pred_part_005)
        labels_pred_010.append(labels_pred_part_010)
        labels_pred_025.append(labels_pred_part_025)
        labels_pred_050.append(labels_pred_part_050)
        labels_pred_075.append(labels_pred_part_075)
        labels_pred_100.append(labels_pred_part_100)
        print(labels_pred_005)
        print(labels_pred_010)
        print(labels_pred_025)
        print(labels_pred_050)
        print(labels_pred_075)
        print(labels_pred_100)
    print(labels_pred_005)
    print(labels_pred_010)
    print(labels_pred_025)
    print(labels_pred_050)
    print(labels_pred_075)
    print(labels_pred_100)
    torch.save(labels_pred_005, "llabel_0.05_dis.pt")
    torch.save(labels_pred_010, "llabel_0.1_dis.pt")
    torch.save(labels_pred_025, "llabel_0.25_dis.pt")
    torch.save(labels_pred_050, "llabel_0.5_dis.pt")
    torch.save(labels_pred_075, "llabel_0.75_dis.pt")
    torch.save(labels_pred_100, "llabel_1.0_dis.pt")
    
    torch.save(blk_number_005, "bblk_number_005.pt")
    torch.save(blk_number_010, "bblk_number_010.pt")
    torch.save(blk_number_025, "bblk_number_025.pt")
    torch.save(blk_number_050, "bblk_number_050.pt")
    torch.save(blk_number_075, "bblk_number_075.pt")
    torch.save(blk_number_100, "bblk_number_100.pt")

def make_one_clustering(load_name, save_name):
    weights = torch.load(load_name)
    st = [0]
    ed = [157711]
    blk_number = [30]
    labels_pred = []
    cur_block_number = 0
    cnt = 1
    for part in range(0, cnt):
        for i in range(st[part], ed[part] + 1):
            labels_pred.append(part + 1)
    labels_pred = torch.tensor(labels_pred)
    print(labels_pred)
    print(len(labels_pred))
    torch.save(labels_pred, save_name)

def make_three_clustering(load_name, save_name):
    weights = torch.load(load_name)
    st = [0, 57669, 80909]
    ed = [57668, 80908, 157711]
    blk_number = [11, 4, 15]
    labels_pred = []
    cur_block_number = 0
    cnt = 3
    for part in range(0, cnt):
        for i in range(st[part], ed[part] + 1):
            labels_pred.append(part + 1)
    labels_pred = torch.tensor(labels_pred)
    print(labels_pred)
    print(len(labels_pred))
    torch.save(labels_pred, save_name)

def make():
    load_name = ["total_global_llava158k_hidden_state.pt"]
    save_name = ["total_clustering_global_llava158k_hidden_state.pt"]
    for i in range(0, len(save_name)):
        make_clustering(load_name[i], save_name[i])

def make_split():
    load_name = ["total_global_llava158k_hidden_state.pt"]
    save_name = ["total_clustering_global_llava158k_hidden_state_split.pt"]
    for i in range(0, len(save_name)):
        make_split_clustering(load_name[i], save_name[i])

def make_one():
    load_name = ["total_global_llava158k_hidden_state.pt"]
    save_name = ["total_clustering_global_llava158k_hidden_state_one.pt"]
    for i in range(0, len(save_name)):
        make_one_clustering(load_name[i], save_name[i])
    
def make_three():
    load_name = ["total_global_llava158k_hidden_state.pt"]
    save_name = ["total_clustering_global_llava158k_hidden_state_three.pt"]
    for i in range(0, len(save_name)):
        make_three_clustering(load_name[i], save_name[i])

def make_665k_split():
    load_name = ["logits.pt"]
    save_name = ["total_clustering_global_llava665k_hidden_state_split_1000.pt"]
    for i in range(0, len(save_name)):
        make_split_665k_clustering(load_name[i], save_name[i])

if __name__ == '__main__':

#    make()
#    make_split()
#    make_one()
#    make_three()
    make_665k_split()
