import math
import copy

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import einops

class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim


        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.b_out = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
 
        batch_size, time_step, input_dim = input.size() # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:,-1,:]) # b h
        input_k = self.W_k(input)# b t h
        input_v = self.W_v(input)# b t h

        if self.attention_type == 'add': #B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            h = q + input_k + self.b_in # b t h
            h = self.tanh(h) #B*T*H
            e = self.W_out(h) # b t 1
            e = torch.reshape(e, (batch_size, time_step))# b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1)) #B*h 1
            e = torch.matmul(input_k, q).squeeze(-1)#b t
            
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1,time_step,1)# b t h
            k = input_k
            c = torch.cat((q, k), dim=-1) #B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        a = self.softmax(e) #B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze() #B*I

        return v, a


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



# Multi-Relational Graph Update, returns a adjacency matrix and Clustering Info
def GraphUpdate(sim_metric, feature_emb, input_dim, n_clu, feat2clu=None):
    adj_mat = torch.zeros(input_dim+1, input_dim+1)
    eps = 1e-7
    #print(feature_emb.size())

    if sim_metric == 'euclidean':
        feature_mean_emb = [None for i in range(input_dim)]
        for i in range(input_dim):
            feature_mean_emb[i] = torch.mean(feature_emb[:,i,:].squeeze(), dim=0).cpu().numpy()
        feature_mean_emb = np.array(feature_mean_emb)
        #print(feature_mean_emb.shape)
        
        if feat2clu is None:
            kmeans = KMeans(n_clusters=n_clu, init='random', n_init=2).fit(feature_mean_emb)
            feat2clu = kmeans.labels_
        
        clu2feat = [[] for i in range(n_clu)]
        for i in range(input_dim):
            clu2feat[feat2clu[i]].append(i)

        for clu_id, cur_clu in enumerate(clu2feat):
            for i in cur_clu:
                for j in cur_clu:
                    if i != j:
                        cos_sim = np.dot(feature_mean_emb[i], feature_mean_emb[j])
                        cos_sim = cos_sim / max(eps, float(np.linalg.norm(feature_mean_emb[i]) * np.linalg.norm(feature_mean_emb[j])))
                        adj_mat[i][j] = torch.tensor(cos_sim)


    elif 'kernel' in sim_metric:
        kernel_mat = torch.zeros((input_dim, input_dim))
        sigma = 0
        for i in range(input_dim):
            for j in range(input_dim):
                if sim_metric == 'rbf_kernel':
                    sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=2)
                if sim_metric == 'laplacian_kernel':
                    sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=1)
                sigma += torch.mean(sample_dist)
        
        sigma = sigma / (input_dim * input_dim)
        #sigma = feature_emb.size(-1)
    
        for i in range(input_dim):
            for j in range(input_dim):
                if sim_metric == 'rbf_kernel':
                    sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=2)
                    kernel_mat[i, j] = torch.mean(torch.exp(-(sample_dist * sample_dist) / (2 * (sigma**2))))
                elif sim_metric == 'laplacian_kernel':
                    sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=1)
                    kernel_mat[i, j] = torch.mean(torch.exp(-sample_dist / sigma))
        #print(kernel_mat)
        aff_mat = np.array(kernel_mat.cpu().detach().numpy())
        #print(aff_mat)
        
        if feat2clu is None:
            kmeans = SpectralClustering(n_clusters=n_clu, affinity='precomputed', n_init=20).fit(aff_mat)
            feat2clu = kmeans.labels_
        
        clu2feat = [[] for i in range(n_clu)]
        for i in range(input_dim):
            clu2feat[feat2clu[i]].append(i)

        for clu_id, cur_clu in enumerate(clu2feat):
            for i in cur_clu:
                for j in cur_clu:
                    if i != j:
                        adj_mat[i][j] = torch.tensor(aff_mat[i][j])


    for i in range(input_dim + 1):
        adj_mat[i][i] = 1

    for i in range(input_dim):
        adj_mat[i][input_dim] = 1
        adj_mat[input_dim][i] = 1

    
    return adj_mat, feat2clu, clu2feat

class MCGRU(nn.Module):
    """
    input: x -> [bs, ts, lab_dim]
    output: [bs, ts, n_feature, feat_dim]
    """
    def __init__(self, feat_dim: int=8, **kwargs):
        super().__init__()
        # mimic-iv
        # self.num_features = 17 # 12 lab test + 5 categorical features
        # self.dim_list = [2,8,12,13,12,1,1,1,1,1,1,1,1,1,1,1,1]
        # cdsl
        self.num_features = 73
        self.dim_list = [1 for _ in range(73)]
        self.feat_dim = feat_dim
        self.grus = nn.ModuleList(
            [
                nn.GRU(dim, feat_dim, num_layers=1, batch_first=True)
                for dim in self.dim_list
            ]
        )
    def forward(self, x):
        # for each feature, apply gru
        bs, ts, lab_dim = x.shape
        out = torch.zeros(bs, ts, self.num_features, self.feat_dim).to(x.device)
        # each feature's dim is different, as in the dim_list, so we need to iterate over it
        # the dim is the channel dim of each feature
        for i, gru in enumerate(self.grus):
            start_pos = sum(self.dim_list[:i])
            end_pos = sum(self.dim_list[:i+1])
            # print(start_pos, end_pos)
            cur_feat = x[:, :, start_pos:end_pos]
            # print(cur_feat.shape)
            cur_feat = gru(cur_feat)[0]
            out[:, :, i] = cur_feat
        return out

class SAFARI(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_clu=8, keep_prob=0.5, **kwargs):
        super(SAFARI, self).__init__()

        # hyperparameters
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim  # d_model
        self.keep_prob = keep_prob
        self.n_clu = n_clu
        
        self.mcgru = MCGRU(feat_dim=hidden_dim)
        self.feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',dropout = 1 - self.keep_prob)

        self.GCN_W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.GCN_W2 = nn.Linear(self.hidden_dim, self.hidden_dim)

#         self.demo_proj_main = nn.Linear(12, self.hidden_dim)
        self.demo_proj = nn.Linear(2, self.hidden_dim)
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # mimic-iv
        # self.proj = nn.Linear(48*self.hidden_dim, self.hidden_dim)
        # cdsl
        self.proj = nn.Linear(76*self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.elu=nn.ELU()

    def forward(self, input, static, **kwargs):
        # adj_mat = torch.randn(18, 18).to(input.device)

        # GRU_embeded_input = self.mcgru(input).mean(dim=1)
        # print(input.shape, static.shape)
        x = self.mcgru(input)
        # print(x.shape)
        x = einops.rearrange(x, 'b t d f -> b d (t f)')
        # print(x.shape)
        x = self.proj(x)
        GRU_embeded_input = x

        static_emb = self.feature_proj(self.relu(self.demo_proj(static))).unsqueeze(1)
        
        GRU_embeded_input = torch.cat((GRU_embeded_input, static_emb), dim=1)
        posi_input = self.dropout(GRU_embeded_input) # batch_size * d_input * hidden_dim

        contexts = posi_input
        
        # clu_context = None
        # gcn_hidden = None
        # gcn_contexts = None
        # #Graph Conv
        # if gcn_hidden is None:
        #     gcn_hidden = self.relu(self.GCN_W1(torch.matmul(adj_mat, contexts)))
        # if gcn_contexts is None:
        #     gcn_contexts = self.relu(self.GCN_W2(torch.matmul(adj_mat, gcn_hidden)))
        

        # clu_context = gcn_contexts[:,:,:]

        clu_context = contexts
        weighted_contexts = self.FinalAttentionQKV(clu_context)[0]
        output = self.relu(self.output0(self.dropout(weighted_contexts)))
          
        return output
