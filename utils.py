import numpy as np
import torch
import random
import math
import os
import scipy.sparse as sp
from torch.nn.functional import softmax

t_mut = 2 #所有组最大真数与取的总样本数的倍数

def get_feature(label_list, data, node_max):
    node_set = set()
    node_label_dict = dict()
    for s, t, l, s_label, t_label in data:
        node_set.update([s, t])
        if(s not in node_label_dict):
            node_label_dict[s] = s_label
        if(t not in node_label_dict):
            node_label_dict[t] = t_label
    
    # total_edge = len(data)
    node_num = len(node_set)
    features = torch.zeros(node_max, max(len(label_list), 16))
    node_label_list = []
    for i in node_set:
        node_label_list.append(node_label_dict[i])
    label_index = []
    for i in node_label_list:
        label_index.append(label_list.index(i))
    features.scatter_(dim=1, index = (torch.LongTensor(label_index).view(-1,1)), value=1)
    mask = torch.zeros(node_max, 1)
    # stdv = 1. / math.sqrt(total_edge)
    # features[:node_num].data.uniform_(-stdv, stdv)
    mask[:node_num] = 1
    
    return features.type(torch.FloatTensor), mask.type(torch.FloatTensor), node_num


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) 
    mx = r_mat_inv.dot(mx)

    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx): 
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(data_name = 'aids'):
    features = torch.FloatTensor()
    masks = torch.FloatTensor()
    graphs = []
    query_node_num = []
    data_node_num = []
    node_max = 0
    querys = os.listdir('../dataset/result_{}/query'.format(data_name))
    datas = os.listdir('../dataset/result_{}/data'.format(data_name))
    querys.sort(key= lambda x:int(x[:-4]))
    datas.sort(key= lambda x:int(x[:-4]))
    query_num = len(querys)
    label_set = set()
    for i in querys:
        query = np.genfromtxt("../dataset/result_{}/query/{}".format(data_name, i), dtype=np.int32, delimiter= ' ')
        label_set.update(query[:,3])
        label_set.update(query[:,4])
    for i in datas:
        data = np.genfromtxt("../dataset/result_{}/data/{}".format(data_name, i), dtype=np.int32, delimiter= ' ')
        label_set.update(data[:,3])
        label_set.update(data[:,4])
    label_list = list(label_set)
    for i in querys:
        query = np.genfromtxt("../dataset/result_{}/query/{}".format(data_name, i), dtype=np.int32, delimiter= ' ')
        node_max = max(node_max, query[:,:2].max())
        graphs.append(query)
    node_max += 1
    for i in range(len(querys)):
        feature, mask, node_num = get_feature(label_list, graphs[i], node_max)
        query_node_num.append(node_num)
        features = torch.cat([features, feature], dim = 0)
        masks = torch.cat([masks, mask], dim = 0)
        graphs[i][:,:2] += (node_max * i)
    # try:
    #     query_label = np.genfromtxt("../dataset/result_{}/data_sub_result.txt".format(data_name))
    # except:
    #     query_label = np.genfromtxt("../dataset/result_{}/new_data_sub_result.txt".format(data_name))
    query_label = np.genfromtxt("../dataset/result_{}/new_data_sub_result.txt".format(data_name))
    query_label = query_label.T
    one_idx_list = []
    train_idx_list = []
    ground_truth_list = []
    max_true_num = 0
    min_true_num = 1000
    one_nums = 0
    one_len_list = []
    for idx, i in enumerate(query_label):
        one_idx = np.argwhere(i == 1).squeeze()
        np.random.shuffle(one_idx)
        one_nums += len(one_idx)
        one_len_list.append(len(one_idx))
        one_idx_list.append(one_idx.tolist())
        max_true_num = max(max_true_num, len(one_idx))
        min_true_num = min(min_true_num, len(one_idx))
    total_num = int(one_nums/100 * 3)

    all_one_num = 0
    for one_len in one_len_list:
        if(one_len > total_num):
            all_one_num += total_num
        else:
            all_one_num += one_len
    last_sort_num = total_num * 100 - all_one_num
    extra_num = 0
    last_extra_num = 0
    while(1):
        extra_num += 1
        all_one_num = 0
        for one_len in one_len_list:
            if(one_len > (total_num + extra_num)):
                all_one_num += (total_num + extra_num)
            else:
                all_one_num += one_len
        tmp_short_num = (total_num + extra_num) * 100 - all_one_num
        if(tmp_short_num < last_sort_num):
            last_sort_num = tmp_short_num
            last_extra_num = extra_num
        else:
            break
    total_num += last_extra_num
    # total_num = int(max_true_num + min_true_num)
    for idx, j in enumerate(one_idx_list):
        train_idx_list.append([(np.ones(total_num).astype(int) * idx).tolist()])
        # train_idx_list.append([idx,])
        true_len = len(j)
        if(true_len > total_num):
            true_len = total_num
            train_idx_list[-1].append(j[:total_num])
            ground_truth_list.append(np.ones(total_num).astype(int).tolist())
        else:
            train_idx_list[-1].append(j)
            ground_truth_list.append(np.ones(len(j)).astype(int).tolist())
            
        
    for idx, i in enumerate(query_label):
        zero_idx = np.argwhere(i == 0).squeeze()
        idx_len = total_num - len(train_idx_list[idx][-1])
        zero_idx_list = np.random.choice(zero_idx, size=idx_len, replace=False, p=None)
        ground_truth_list[idx].extend(np.zeros(idx_len).astype(int).tolist())
        train_idx_list[idx][-1].extend(zero_idx_list)

    for i in datas:
        data = np.genfromtxt("../dataset/result_{}/data/{}".format(data_name, i), dtype=np.int32, delimiter= ' ')
        feature, mask, node_num = get_feature(label_list, data, node_max)
        data_node_num.append(node_num)
        features = torch.cat([features, feature], dim = 0)
        masks = torch.cat([masks, mask], dim = 0)
        data[:,:2] += (node_max * (int(i[:-4]) + query_num))
        graphs.append(data)
    tmp_datas = graphs[0][:,:3]
    for i in graphs[1:]:
        tmp_datas = np.concatenate((tmp_datas, i[:,:3]),axis=0)
    # adj = sp.coo_matrix((np.ones(tmp_datas.shape[0]), 
    #                 (tmp_datas[:, 0], tmp_datas[:, 1])),
    #                     shape=(features.shape[0], features.shape[0]),
    #                     dtype=np.float32)
    adj = sp.coo_matrix((tmp_datas[:,2], 
                    (tmp_datas[:, 0], tmp_datas[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0])) 
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, query_num, torch.LongTensor(ground_truth_list), adj, node_max, masks, torch.LongTensor(train_idx_list), torch.LongTensor(query_node_num), torch.LongTensor(data_node_num)