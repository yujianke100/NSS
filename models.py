import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphNN


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(GCN, self).__init__() 
        self.gc1 = GraphConvolution(nfeat, nhid) 
        self.gc2 = GraphConvolution(nhid, nclass) 
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.tanh(x)
        return x

class NN(nn.Module):
    def __init__(self, nfeat, nhid): 
        super(NN, self).__init__()
        self.fnn = GraphNN(nfeat, nhid)
    
    def forward(self, x):
        return F.relu(self.fnn(x))

class RENN(nn.Module):
    def __init__(self, nclass, nfeat):
        super(RENN, self).__init__()
        self.renn = GraphNN(nclass, nfeat)

    def forward(self, x):
        return torch.softmax(self.renn(x),dim=-1)
        # return torch.tanh(self.renn(x))

class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=64):
        super(WDiscriminator, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        # self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)
    def forward(self, input_embd):
        # return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))
        return F.leaky_relu(self.output(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True)

class transformation(torch.nn.Module):
    def __init__(self, hidden_size):
        super(transformation, self).__init__()
        self.trans = torch.nn.Parameter(torch.eye(hidden_size))
    def forward(self, input_embd):
        return input_embd.matmul(self.trans)