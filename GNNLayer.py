import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from opt import args

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

class GAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, n_z):
        super(GAE, self).__init__()

        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.gnn_4 = GNNLayer(gae_n_enc_3, n_z)

        self.gnn_5 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_6 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_7 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_8 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=False if args.name == "hhar" else True)
        z = self.gnn_2(z, adj, active=False if args.name == "hhar" else True)
        z = self.gnn_3(z, adj, active=False if args.name == "hhar" else True)
        z_gae = self.gnn_4(z, adj, active=False)
        z_gae_adj = self.s(torch.mm(z_gae, z_gae.t()))

        z = self.gnn_5(z_gae, adj, active=False if args.name == "hhar" else True)
        z = self.gnn_6(z, adj, active=False if args.name == "hhar" else True)
        z = self.gnn_7(z, adj, active=False if args.name == "hhar" else True)
        z_hat = self.gnn_8(z, adj, active=False if args.name == "hhar" else True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        adj_hat = z_gae_adj + z_hat_adj

        return z_gae, z_hat, adj_hat