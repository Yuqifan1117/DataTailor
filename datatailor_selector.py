import torch
import json
import math
import csv
import random
import numpy as np

def final_select():
    ratio = 0.15
    num = 665298
    mix_st = [0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610]
    mix_ed = [57668, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297]
    block_number = len(mix_st)
    final_num_cho = []
    sum = 0.
    informativeness = torch.load("informativeness.pt")
    tau = torch.load("tau.pt")

    for i in range(0, block_number):
        mm = torch.mean(tau[mix_st[i]:(mix_ed[i] + 1)]) * torch.mean(tau[mix_st[i]:(mix_ed[i] + 1)]) 
        chose = (mix_ed[i] - mix_st[i] + 1) * mm
        sum += chose
    one = ratio * num / sum
    for i in range(0, block_number):
        mm = torch.mean(tau[mix_st[i]:(mix_ed[i] + 1)]) * torch.mean(tau[mix_st[i]:(mix_ed[i] + 1)]) 
        chose = (mix_ed[i] - mix_st[i] + 1) * mm
        select = one * chose
        final_num_cho.append(select)

    weight1 = []
    weight_part = []
    weight_block = []
    length = []

    labels_pred = torch.tensor([0] * num)
    for i in range(0, block_number):
        for j in range(mix_st[i], (mix_ed[i] + 1)):
            labels_pred[j] = i
    
    _res = []
    print("Start init")
    for i in range(0, block_number):
        __res = (labels_pred == i).nonzero(as_tuple=True)[0].tolist()
        _res.append(__res)

    representativeness = torch.load("representativeness.pt")
    uniqueness = torch.load("uniqueness.pt")
    
    json_root=r'llava_v1_5_mix665k.json'
    with open(json_root, 'r') as f:
        json_info = json.load(f)
        
    print("Start select") 
    
    selected_DATATAILOR = []
    
    R = torch.tensor([0] * num)
    for i in range(0, num):
        R[i] = len(json_info[i]["conversations"]) / 2
    for data_type in range(0, block_number):
        pair_DATATAILOR = []
        total = int(final_num_cho[data_type])   
        i = data_type
        informativeness_data = informativeness[mix_st[i]:(mix_ed[i]+1)]
        uniqueness_data = uniqueness[mix_st[i]:(mix_ed[i]+1)]
        representativeness_data = representativeness[mix_st[i]:(mix_ed[i]+1)]
        informativeness_data = np.argsort(np.argsort(informativeness_data)) / (len(informativeness_data))
        uniqueness_data = np.argsort(np.argsort(uniqueness_data)) / (len(uniqueness_data))
        representativeness_data = np.argsort(np.argsort(representativeness_data)) / (len(representativeness_data))
        for pos_i in range(0, len(_res[data_type])):
            i = _res[data_type][pos_i]
            pair_DATATAILOR.append([(informativeness_data[pos_i] * R[i]
                                     + uniqueness_data[pos_i]
                                     + representativeness_data[pos_i])
                                  /  (R[i] + 2)
                                  ,  i])        

        pair_DATATAILOR = sorted(pair_DATATAILOR, key=lambda tup: tup[0], reverse=True)
        for i in range(0, total):
            selected_DATATAILOR.append(pair_DATATAILOR[i][1])

    selected_DATATAILOR = sorted(selected_DATATAILOR)
    print(len(selected_DATATAILOR))

    json_root=r'llava_v1_5_mix665k.json'
    with open(json_root, 'r') as f:
        json_info = json.load(f)
    save_DATATAILOR=[json_info[i] for i in selected_DATATAILOR]

    with open("llava_v1_5_mix665k_0.15_final.json", 'w') as fp:
        json.dump(save_DATATAILOR, fp, indent=4)

if __name__ == '__main__':
    final_select()
    