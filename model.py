
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import copy
def lh_contrast(X_l, X_l_prime, X_h, tau=0.1):
    """
    X_l: [N, D]  - anchor (low freq)
    X_l_prime: [N, D] - positive view (dropout of low freq)
    X_h: [N, D]  - high freq used as hard negatives
    """
    N, D = X_l.shape

    # normalize for cosine similarity
    X_l = F.normalize(X_l, dim=1)
    X_l_prime = F.normalize(X_l_prime, dim=1)
    X_h = F.normalize(X_h, dim=1)

    # similarity scores: [N, N]
    sim_pos = torch.exp(torch.sum(X_l * X_l_prime, dim=1) / tau)      # [N]
    sim_low = torch.exp(torch.mm(X_l, X_l_prime.t()) / tau)           # [N, N]
    sim_high = torch.exp(torch.mm(X_l, X_h.t()) / tau)                # [N, N]

    # exclude diagonal positive from sim_low denominator
    mask = torch.eye(N, dtype=torch.bool, device=X_l.device)
    sim_low = sim_low.masked_fill(mask, 0.0)

    denom = sim_low.sum(dim=1) + sim_high.sum(dim=1) + sim_pos  # [N]
    loss = -torch.log(sim_pos / (denom + 1e-8))
    return loss.mean()

def make_view_feature_dropout(x_input, drop_p=0.2):
    # 直接在输入特征层面做随机丢弃
    mask = (torch.rand_like(x_input) > drop_p).float()
    return x_input * mask

def Contrast_hl(anchor, positive, negative,margin = 1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)

    loss = torch.mean(F.relu(distance_positive - distance_negative + margin))
    return loss

def get_mask(adj):
    if adj.is_sparse:
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj
    mask = (adj_dense > 0).unsqueeze(0)  # (1, N, N)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, att_dropout,num_heads=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dropout_rate = att_dropout
        self.mha_norm = nn.LayerNorm(in_features)
        if in_features != out_features:
            self.proj = nn.Linear(in_features, out_features)
        else:
            self.proj = nn.Identity()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.W_q = Parameter(torch.FloatTensor(in_features, out_features))
        self.W_k = Parameter(torch.FloatTensor(in_features, out_features))
        self.W_v = Parameter(torch.FloatTensor(in_features, out_features))
        self.W_o = Parameter(torch.FloatTensor(out_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, q, k, v, mask = None):
        N, _ = q.shape
        Q = torch.matmul(q, self.W_q)
        K = torch.matmul(k, self.W_k)
        V = torch.matmul(v, self.W_v)
        Q = Q.view(N, self.num_heads, self.head_dim).permute(1, 0, 2)  # (num_heads, N, head_dim)
        K = K.view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = V.view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        scores = torch.einsum('hid,hjd->hij', Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))#float('-inf')
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout_rate, training=self.training)
        h = torch.einsum('hij,hjd->hid', attn, V)  # (num_heads, N, head_dim)
        h = h.permute(1, 0, 2).reshape(N, self.out_features)
        out = torch.matmul(h, self.W_o)
        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, ffn_hidden, dropout ):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, ffn_hidden)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(ffn_hidden, in_features)
        self.dropput = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropput(x)
        x = self.linear_2(x)
        return x

class GT_Encoder(nn.Module):
    def __init__(self, args, in_feats,out_feats, dropout):
        super().__init__()
        self.args = args
        self.mha_norm = nn.LayerNorm(out_feats)
        self.ffn_norm = nn.LayerNorm(out_feats)
        # self.effectattn = EfficientAttention(in_channels = d_model, key_channels =d_model, head_count =heads, value_channels = d_model)
        self.mha = MultiHeadAttention(in_features=in_feats, out_features=out_feats,att_dropout = args.dropout_att)
        self.ffn = FeedForwardNetwork(in_features=out_feats,ffn_hidden=out_feats, dropout=args.dropout_att)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj = None):
        x = self.mha_norm(x)
        att = self.mha(x, x, x, adj)
        x = x + att
        x2 = self.ffn_norm(x)
        x = x + self.dropout(self.ffn(x2))
        return x

class GADGT_model(nn.Module):
    def __init__(self, arg,feat_size):
        super(GADGT_model, self).__init__()
        self.arg= arg
        self.nclass = arg.nb_classes
        self.hid_dim = arg.hidden_dim
        self.dropout = arg.dropout
        self.Trans_layer_num = arg.Trans_layer_num
        self.layers_low = get_clones(GT_Encoder(arg, arg.hidden_dim,arg.hidden_dim, arg.dropout_att), arg.Trans_layer_num)
        self.layers_high = get_clones(GT_Encoder(arg, arg.hidden_dim,arg.hidden_dim, arg.dropout_att), arg.Trans_layer_num)
        self.MLP1 = nn.Linear(feat_size, self.hid_dim)
        self.MLP2 = nn.Linear(feat_size, self.hid_dim)
        self.act = nn.ReLU()
        self.linear  = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, self.nclass)
        self.lamb = nn.Parameter(torch.zeros(2), requires_grad=True)


    def forward(self, x_input, adj):
        X_l = self.MLP1(x_input)
        X_h = self.MLP2(x_input)
        for i in range(self.Trans_layer_num):
            # X_l = self.layers_low[i](X_l, adj)
            X_l = self.layers_low[i](X_l)
            X_h = self.layers_high[i](X_h)
        view2_input = make_view_feature_dropout(x_input, drop_p=0.2)  # 或 make_view_gaussian
        X_l_prime = self.MLP1(view2_input)
        for i in range(self.Trans_layer_num):
            # X_l_prime = self.layers_low[i](X_l_prime, adj)
            X_l_prime = self.layers_low[i](X_l_prime)
        C_loss= lh_contrast(X_l,X_l_prime,X_h)
        # X_out =  X_l * self.lamb[0] + X_h * (1. + self.lamb[1])
        X_out = X_l * self.lamb[0] + X_h * (1. + self.lamb[1])
        X_out = self.linear(X_out)
        X_out = self.act(X_out)
        X_out = self.linear2(X_out)

        return X_out, C_loss

