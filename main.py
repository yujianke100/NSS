from __future__ import division
from __future__ import print_function
from platform import node

import time
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import itertools

from utils import load_data
from models import *
from os import mkdir
from os.path import exists

# from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='nci',
                    help='data_name, nci, pubchem, fda.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--recon', action='store_true', default=True,
                    help='Enable recon.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--re_lr', type=float, default=0.001,
                    help='Initial recon learning rate.')
parser.add_argument('--lr_wd', type=float, default=0.01)
parser.add_argument('--predict_th', type=float, default=0.65)
parser.add_argument('--emb_size', type=int, default=400,
                    help='Number of emb_size to train.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--gcn_out', type=int, default=64,
                    help='Number of out units.')
# parser.add_argument('--out', type=int, default=32,
#                     help='Number of out units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

def w_loss(q_nodes_features, d_nodes_features):
    q_nodes_features = q_nodes_features.reshape(q_nodes_features.shape[0]*q_nodes_features.shape[1], q_nodes_features.shape[2])
    d_nodes_features = d_nodes_features.reshape(d_nodes_features.shape[0]*d_nodes_features.shape[1], d_nodes_features.shape[2])
    for j in range(5):
        w0 = wdiscriminator(q_nodes_features)
        w1 = wdiscriminator(d_nodes_features)
        anchor1 = w1.view(-1).argsort(descending=True)[: d_nodes_features.size(0)]
        anchor0 = w0.view(-1).argsort(descending=False)[: d_nodes_features.size(0)]
        embd0_anchor = q_nodes_features[anchor0, :].clone().detach()
        embd1_anchor = d_nodes_features[anchor1, :].clone().detach()
        optimizer_wd.zero_grad()
        loss = -torch.mean(wdiscriminator(embd0_anchor)) + torch.mean(wdiscriminator(embd1_anchor))
        loss.backward()
        optimizer_wd.step()
        for p in wdiscriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
    w0 = wdiscriminator(q_nodes_features)
    w1 = wdiscriminator(d_nodes_features)
    anchor1 = w1.view(-1).argsort(descending=True)[: d_nodes_features.size(0)]
    anchor0 = w0.view(-1).argsort(descending=False)[: d_nodes_features.size(0)]
    embd0_anchor = q_nodes_features[anchor0, :]
    embd1_anchor = d_nodes_features[anchor1, :]
    return -torch.mean(wdiscriminator(embd1_anchor))

def train(query_num, ground_truth, adjs, total_num, node_max, masks, train_idx_set_list, t_q_num, t_d_num, train_data_idx, train_node_num):
    model.train()
    extract_model.train()
    trans.train()
    wdiscriminator.train()

    optimizer.zero_grad()
    optimizer_wd.zero_grad()
    if(args.recon):
        renn_model.train()
        renn_optimizer.zero_grad()
    
    extract_features = extract_model(features) * masks
    output = model(extract_features, adjs) * masks
    if(args.recon):
        recon = renn_model(output) * masks
        recon = recon[train_idx_set_list]
    output = output.reshape(total_num, node_max, output.shape[-1])
    
    comp_features = features * masks
    comp_features = comp_features[train_idx_set_list]

    q_nodes_features = output[:int(query_num * 0.8)]
    d_nodes_features = output[query_num:]

    l_w = w_loss(q_nodes_features, trans(d_nodes_features))

    d_nodes_features = d_nodes_features[train_data_idx]
    d_nodes_features = d_nodes_features.reshape(t_q_num , t_d_num, d_nodes_features.shape[1], d_nodes_features.shape[2])
    relation =  torch.matmul(q_nodes_features.unsqueeze(1), d_nodes_features.permute(0,1,3,2))
    relation_sum = relation.sum(dim=-1).sum(dim=-1)
    avg_relationt = relation_sum / train_node_num
    results = torch.sigmoid(avg_relationt)
    predict = torch.where(results > args.predict_th, 1, 0)
    l_m = ( - (ground_truth * torch.log(results + 1e-9) + (1 - ground_truth) * torch.log(1 - results + 1e-9))).mean()
    if(args.recon):
        l_r = torch.norm(recon - comp_features, dim=1, p=2).mean()
        loss_train = l_m * 0.9895 + l_r * 0.01 + l_w * 0.0005
    else:
        loss_train = l_m * 0.999 + l_w * 0.001
    loss_train.backward() 
    optimizer.step()
    optimizer_wd.step()
    if(args.recon):
        renn_optimizer.step()
    ground_truth = ground_truth.reshape(ground_truth.shape[0] * ground_truth.shape[1])
    predict = predict.reshape(predict.shape[0] * predict.shape[1])
    ground_truth = ground_truth.cpu()
    predict = predict.cpu()
    # print('loss: {:.4f}'.format(loss_train))
    #     #   'acc:{:.4f}'.format(accuracy_score(ground_truth, predict)),
    #     #   "pre:{:.4f}".format(precision_score(ground_truth, predict)),
    #     #   "F1:{:.4f}".format(f1_score(ground_truth, predict)),
    #     #   "recall:{:.4f}".format(recall_score(ground_truth, predict)))

def test(query_num, ground_truth, adjs, total_num, node_max, masks, test_idx_set_list, t_q_num, t_d_num, test_data_idx, test_node_num):

    model.eval()
    extract_model.eval()
    extract_features = extract_model(features) * masks
    output = model(extract_features, adjs) * masks

    output = output.reshape(total_num, node_max, output.shape[-1])
    comp_features = features * masks
    comp_features = comp_features[test_idx_set_list]
    
    d_nodes_features = output[query_num:][test_data_idx]
    d_nodes_features = d_nodes_features.reshape(t_q_num , t_d_num, d_nodes_features.shape[1], d_nodes_features.shape[2]).permute(0,1,3,2)
    start_time = time.time()
    relation =  torch.matmul(q_nodes_features, d_nodes_features).sum(dim=-1).sum(dim=-1) / test_node_num
    results = torch.sigmoid(relation)
    predict = torch.where(results > args.predict_th, 1, 0)
    end_time = time.time()
    print("Test set results:",
        #   "loss= {:.4f}".format(loss_test.item()),
          "time:= {:.6f}".format(end_time - start_time))
    ground_truth = ground_truth.reshape(int(ground_truth.shape[0] * ground_truth.shape[1]))
    predict = predict.reshape(predict.shape[0] * predict.shape[1])
    ground_truth = ground_truth.cpu()
    predict = predict.cpu()
    print('accuracy:{:.4f}'.format(accuracy_score(ground_truth, predict)))
    print("prediction:{:.4f}".format(precision_score(ground_truth, predict)))
    print("F1-Score:{:.4f}".format(f1_score(ground_truth, predict)))
    print("recall-score:{:.4f}".format(recall_score(ground_truth, predict)))

def start_train(query_num, ground_truth, adjs, node_max, masks, train_idx_list, query_node_num, data_node_num):
    global features, model, renn_model, extract_model, trans, wdiscriminator
    q_idx_list = train_idx_list[:int(query_num*0.8)]
    t_idx_list = train_idx_list[int(query_num*0.8):]
    train_idx_set_list = torch.LongTensor(list(set(q_idx_list[:,0][:,0].tolist() + (q_idx_list[:,1].reshape(q_idx_list[:,1].shape[0]*q_idx_list[:,1].shape[1])+query_num).tolist())))
    test_idx_set_list = torch.LongTensor(list(set(t_idx_list[:,0][:,0].tolist() + (t_idx_list[:,1].reshape(t_idx_list[:,1].shape[0]*t_idx_list[:,1].shape[1])+query_num).tolist())))
    if args.cuda:
        model.cuda() 
        extract_model.cuda()
        trans.cuda()
        wdiscriminator.cuda()
        if(args.recon):
            renn_model.cuda()
        features = features.cuda()
        ground_truth, adjs, masks, query_node_num, data_node_num =  ground_truth.cuda(), adjs.cuda(), masks.cuda(), query_node_num.cuda(), data_node_num.cuda()
        q_idx_list, t_idx_list = q_idx_list.cuda(), t_idx_list.cuda()
        train_idx_set_list, test_idx_set_list = train_idx_set_list.cuda(), test_idx_set_list.cuda()
    t_q_num = q_idx_list[:,1].shape[0]
    t_d_num = q_idx_list[:,1].shape[1]
    train_data_idx = q_idx_list[:,1].reshape(t_q_num * t_d_num)
    train_query_node_num = query_node_num[:int(query_num*0.8)]
    train_data_node_num = data_node_num[train_data_idx].reshape(t_q_num, t_d_num)
    train_node_num = train_query_node_num.expand(train_data_node_num.shape[1], train_data_node_num.shape[0]).T * train_data_node_num
    total_num = int(features.shape[0] / node_max)

    t_t_q_num = t_idx_list[:,1].shape[0]
    t_t_d_num = t_idx_list[:,1].shape[1]
    test_data_idx = t_idx_list[:,1].reshape(t_t_q_num * t_t_d_num)
    test_query_node_num = query_node_num[int(query_num*0.8):query_num]
    test_data_node_num = data_node_num[test_data_idx].reshape(t_t_q_num, t_t_d_num)
    test_node_num = test_query_node_num.expand(test_data_node_num.shape[1], test_data_node_num.shape[0]).T * test_data_node_num
    

    if(not exists('./model_{}'.format(data_name))):
        mkdir('./model_{}'.format(data_name))
    # train(query_num, ground_truth[:int(query_num*0.8)], adjs, total_num, node_max, masks, train_idx_set_list, t_q_num, t_d_num, train_data_idx, train_node_num)
    if(not exists('./model_{}/{}_main_model.pt'.format(data_name, data_name))):
        for epoch in range(1, args.epochs+1):
            # print('epoch:{}'.format(epoch))
            train(query_num, ground_truth[:int(query_num*0.8)], adjs, total_num, node_max, masks, train_idx_set_list, t_q_num, t_d_num, train_data_idx, train_node_num)
        torch.save(model, './model_{}/{}_main_model.pt'.format(data_name, data_name))
        torch.save(model.state_dict(),'./model_{}/{}_main_model_states.pt'.format(data_name, data_name))
        torch.save(extract_model, './model_{}/{}_extra_model.pt'.format(data_name, data_name))
        torch.save(extract_model.state_dict(),'./model_{}/{}_extra_model_states.pt'.format(data_name, data_name))
        torch.save(renn_model, './model_{}/{}_renn_model.pt'.format(data_name, data_name))
        torch.save(renn_model.state_dict(),'./model_{}/{}_renn_model_states.pt'.format(data_name, data_name))
        torch.save(trans, './model_{}/{}_trans_model.pt'.format(data_name, data_name))
        torch.save(trans.state_dict(),'./model_{}/{}_trans_model_states.pt'.format(data_name, data_name))
        torch.save(wdiscriminator, './model_{}/{}_wd_model.pt'.format(data_name, data_name))
        torch.save(wdiscriminator.state_dict(),'./model_{}/{}_wd_model_states.pt'.format(data_name, data_name))
    else:
        model.load_state_dict(torch.load('./model_{}/{}_main_model_states.pt'.format(data_name, data_name)))
        extract_model.load_state_dict(torch.load('./model_{}/{}_extra_model_states.pt'.format(data_name, data_name)))
        renn_model.load_state_dict(torch.load('./model_{}/{}_renn_model_states.pt'.format(data_name, data_name)))
        trans.load_state_dict(torch.load('./model_{}/{}_trans_model_states.pt'.format(data_name, data_name)))
        wdiscriminator.load_state_dict(torch.load('./model_{}/{}_wd_model_states.pt'.format(data_name, data_name)))

    test(query_num, ground_truth[int(query_num*0.8):], adjs, total_num, node_max, masks, test_idx_set_list, t_t_q_num, t_t_d_num, test_data_idx, test_node_num)


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    data_name = args.data_name
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    features, query_num, ground_truth, adjs, node_max, masks, train_idx_list, query_node_num, data_node_num = load_data(data_name = data_name)
    model = GCN(nfeat=args.emb_size,
                nhid=args.hidden,
                nclass=args.gcn_out,
                dropout=args.dropout)

    extract_model = NN(nfeat=features.shape[1], nhid=args.emb_size)
    
    renn_model = RENN(nclass=args.gcn_out,nfeat=features.shape[1])

    trans = transformation(args.gcn_out)
                    
    optimizer = optim.Adam(itertools.chain(extract_model.parameters(), model.parameters(), trans.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    
    renn_optimizer = optim.Adam(renn_model.parameters(),
                            lr=args.re_lr, weight_decay=args.weight_decay)

    wdiscriminator = WDiscriminator(args.gcn_out)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.lr_wd, weight_decay=5e-4)

    start_train(query_num, ground_truth, adjs, node_max, masks, train_idx_list, query_node_num, data_node_num)
    print('dataset:',data_name)
    
