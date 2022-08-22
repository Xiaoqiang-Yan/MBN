import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from AE import AE
from GNNLayer import GNNLayer
from opt import args

class MBN(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, v, n_node, device):
        super(MBN, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        ae_pre = 'ae_pretrain/{}.pkl'.format(args.name)
        self.ae.load_state_dict(torch.load(ae_pre, map_location='cpu'))
        print('Loading AE pretrain model:', ae_pre)

        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.gnn_4 = GNNLayer(gae_n_enc_3, n_z)
        self.gae_fc = GNNLayer(n_z, n_clusters)

        self.gnn_5 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_6 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_7 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_8 = GNNLayer(gae_n_dec_3, n_input)

        self.s = nn.Sigmoid()

        self.a = 0.5

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.s = nn.Sigmoid()
        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae, x_bar, enc_h1, enc_h2, enc_h3 = self.ae(x)

        gae_enc1 = self.gnn_1(x, adj, active=True)
        gae_enc2 = self.gnn_2((1 - self.a) * gae_enc1 + self.a * enc_h1, adj, active=True)
        gae_enc3 = self.gnn_3((1 - self.a) * gae_enc2 + self.a * enc_h2, adj, active=True)
        z_gae = self.gnn_4((1 - self.a) * gae_enc3 + self.a * enc_h3, adj, active=False)
        z_i = (1 - self.a) * z_gae + self.a * z_ae

        z_l = torch.spmm(adj, z_i)

        gae_dec1 = self.gnn_5(z_gae, adj, active=True)
        gae_dec2 = self.gnn_6(gae_dec1, adj, active=True)
        gae_dec3 = self.gnn_7(gae_dec2, adj, active=True)

        z_hat = self.gnn_8(gae_dec3, adj, active=True)
        adj_hat = self.s(torch.mm(z_gae, z_gae.t()))

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return x_bar, z_hat, adj_hat, z_ae, q, q1, z_l
