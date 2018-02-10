import torch
import torch.nn as nn
from birnn import BiRNN
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
# use_cuda = False

class BiRNNT(BiRNN):
    def __init__(self, v_size, t_size, u_size, emb_dim_v, emb_dim_t, emb_dim_u, emb_dim_d, hidden_dim,distance):
        super(BiRNNT, self).__init__(v_size, emb_dim_v, hidden_dim)
        self.t_size = t_size
        self.v_size = v_size
        self.emb_dim_t = emb_dim_t
        self.emb_dim_u = emb_dim_u
        self.emb_dim_d = emb_dim_d
        self.embedder_t = nn.Embedding(t_size, self.emb_dim_t, padding_idx=0)
        self.embedder_u = nn.Embedding(u_size,self.emb_dim_u,padding_idx=0)
        #self.embedder_lat = nn.Embedding(l_size,self.emb_dim_l,padding_idx=0)
        #self.embedder_lon = nn.Embedding(l_size,self.embd_dim_l,padding_idx=0)
        self.decoder_dim = self.hidden_dim * 2 + self.emb_dim_t + self.emb_dim_u + self.emb_dim_d
        self.decoder = nn.Linear(self.decoder_dim, self.v_size)
        self.distance = distance
        self.linear_d1 = nn.Linear(v_size,1)
        self.embedder_d2 = nn.Embedding(v_size,self.emb_dim_d,padding_idx = 0)

    def get_embeddeing_u(self, uids, len_long, mask_long_valid):
        uids_strip = uids.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_u = self.embedder_u(uids_strip).view(-1, self.emb_dim_u)
        embedding_u_valid = embedding_u.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_u)).view(-1, self.emb_dim_u)
        return embedding_u_valid

    def get_embedding_t(self, tids_next, len_long, mask_long_valid):
        tids_next_strip = tids_next.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_t = self.embedder_t(tids_next_strip).view(-1, self.emb_dim_t)
        embedding_t_valid = embedding_t.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_t)).view(-1, self.emb_dim_t)
        return embedding_t_valid

    def forward(self, uids, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate):
        mask_long_valid = mask_long.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        mask_optim_valid = (mask_optim if len(mask_evaluate) == 0 else mask_evaluate).index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long).data[0])))).masked_select(mask_long_valid)
        embeddings_t = self.get_embedding_t(tids_next, len_long, mask_long_valid)
        embeddings_u = self.get_embeddeing_u(uids, len_long, mask_long_valid)
        d_score = self.get_scores_d_all(vids_long, len_long, mask_long_valid)
        hiddens_long = self.get_hiddens_long(vids_long, len_long, mask_long_valid)
        hiddens_short = self.get_hiddens_short(vids_short_al, len_short_al,short_cnt)
        hiddens_comb = torch.cat((hiddens_long, hiddens_short, embeddings_t,embeddings_u,d_score), 1)
        mask_optim_expanded = mask_optim_valid.view(-1, 1).expand_as(hiddens_comb)
        hiddens_comb_masked = hiddens_comb.masked_select(mask_optim_expanded).view(-1, self.decoder_dim)
        decoded = self.decoder(hiddens_comb_masked)
        return F.log_softmax(decoded)

    def get_scores_d_all(self,vids_long, len_long, mask_long_valid):
        distance_vids_score = Variable(torch.zeros(self.v_size,1))
        for idx,(len_long_vid,vid) in enumerate(zip(len_long,vids_long)):
            distance_vid_score = self.get_scores_d(vid,len_long_vid)
            distance_vids_score = torch.cat((distance_vids_score,distance_vid_score),1)
        #ds = Variable(torch.FloatTensor(distance_vids_score))
        distance_vids_score = np.delete(distance_vids_score.data.numpy(),0,1)
        distance_vids_score = Variable(torch.LongTensor(distance_vids_score))
        ds = torch.t(distance_vids_score)
        ds_strip = ds.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_d = self.embedder_d2(ds_strip).view(-1, self.emb_dim_d)
        embedding_d_valid = embedding_d.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_d)).view(-1, self.emb_dim_d)
        return embedding_d_valid

    def get_scores_d(self,vid,len_long_vid):
        vid = vid.data.numpy()
        vid_d = np.zeros((self.v_size,self.v_size))
        for i in range(len_long_vid):
            for j in range(len_long_vid):
                vid_d[i][j] = self.get_distance(vid[i],vid[j])
        vid_d = Variable(torch.t(torch.FloatTensor(vid_d)))
        distance_vid_score = self.linear_d1(vid_d)
        return distance_vid_score

    def get_distance(self,v1,v2):
        if v1<v2:
            try:
                d = self.distance[v1-1][v2-v1-1]
            except:
                print v1
                print v2
        elif v1>v2:
            d = self.distance[v2-1][v1-v2-1]
        else:
            d = 0
        return d




