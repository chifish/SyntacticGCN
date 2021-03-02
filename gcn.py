import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

class GraphConvolution(nn.Module):
    """
    syntactic GCN network, proposed in https://www.aclweb.org/anthology/D17-1159/
    """

    def __init__(self, in_features, out_features, num_labels=44, dropout=0.2, gating=True, in_arcs=True, out_arcs=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_labels = num_labels
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.gating = gating
        self.dropout = nn.Dropout(p=dropout)

        self.w_in = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, out_features)) for i in range(num_labels)])
        self.w_out = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, out_features)) for i in range(num_labels)])
        self.w_loop = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.b_in = nn.ParameterList([nn.Parameter(torch.FloatTensor(out_features)) for i in range(num_labels)])
        self.b_out = nn.ParameterList([nn.Parameter(torch.FloatTensor(out_features)) for i in range(num_labels)])
        self.b_loop = nn.Parameter(torch.FloatTensor(out_features))

        if self.gating:
            self.w_in_g = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, 1)) for i in range(num_labels)])
            self.w_out_g = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_features, 1)) for i in range(num_labels)])
            self.w_loop_g = nn.Parameter(torch.FloatTensor(in_features, 1))
            self.b_in_g = nn.ParameterList([nn.Parameter(torch.FloatTensor(1)) for i in range(num_labels)])
            self.b_out_g = nn.ParameterList([nn.Parameter(torch.FloatTensor(1)) for i in range(num_labels)])
            self.b_loop_g = nn.Parameter(torch.FloatTensor(1))

    def forward(self, gcn_inp, adj_mat):

        adj_mat = self.dropout(adj_mat)
        h = torch.matmul(gcn_inp, self.w_loop) #(bs, len, hidden_size) x (hidden_size, hidden_size)
        h = h + self.b_loop

        if self.gating:
            loop_gates = torch.sigmoid(torch.matmul(gcn_inp, self.w_loop_g) + self.b_loop_g)
            h = h * loop_gates

        act_sum = h

        for lbl in range(self.num_labels):
            if self.in_arcs:
                inp_in     = torch.matmul(gcn_inp, self.w_in[lbl]) + self.b_in[lbl] #inp_in: (bs, len, hidden_size)
                if self.gating:
                    inp_in_g = torch.matmul(gcn_inp, self.w_in_g[lbl]) + self.b_in_g[lbl]
                    inp_in = inp_in * torch.sigmoid(inp_in_g)

                adj_matrix = adj_mat[lbl].permute(0,2,1) #(bs, len, len)
                in_act = torch.bmm(adj_matrix, inp_in)  #aggregate  (bs, len, hidden_size)

            if self.out_arcs:
                inp_out    = torch.matmul(gcn_inp, self.w_out[lbl]) + self.b_out[lbl]
                if self.gating:
                    inp_out_g = torch.matmul(gcn_inp, self.w_out_g[lbl]) + self.b_out_g[lbl]
                    inp_out = inp_out * torch.sigmoid(inp_out_g)

                adj_matrix = adj_mat[lbl]
                out_act = torch.bmm(adj_matrix, inp_out)

            act_sum += in_act + out_act

        return F.relu(act_sum)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
