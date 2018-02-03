import torch
import torch.nn as nn
from birnn import BiRNN
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
# use_cuda = False

class BiRNNT(BiRNN):
    def __init__(self, v_size, t_size, u_size, emb_dim_v, emb_dim_t, emb_dim_u, hidden_dim,distance):
        super(BiRNNT, self).__init__(v_size, emb_dim_v, hidden_dim)
        self.t_size = t_size
        self.emb_dim_t = emb_dim_t
        self.emb_dim_u = emb_dim_u
        self.embedder_t = nn.Embedding(t_size, self.emb_dim_t, padding_idx=0)
        self.embedder_u = nn.Embedding(u_size,self.emb_dim_u,padding_idx=0)
        self.embedder_lat = nn.Embeddign(l_size,self.emb_dim_l,padding_idx=0)
        self.embedder_lon = nn.Embedding(l_size,self.embd_dim_l,padding_idx=0)
        self.decoder_dim = self.hidden_dim * 2 + self.emb_dim_t + self.emb_dim_u
        self.decoder = nn.Linear(self.decoder_dim, self.v_size)
        self.distance = distance
        self.linear_d = nn.Embedding(v_size,1)

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
        hiddens_long = self.get_hiddens_long(vids_long, len_long, mask_long_valid)
        hiddens_short = self.get_hiddens_short(vids_short_al, len_short_al,short_cnt)
        hiddens_comb = torch.cat((hiddens_long, hiddens_short, embeddings_t,embeddings_u), 1)
        mask_optim_expanded = mask_optim_valid.view(-1, 1).expand_as(hiddens_comb)
        hiddens_comb_masked = hiddens_comb.masked_select(mask_optim_expanded).view(-1, self.decoder_dim)
        decoded = self.decoder(hiddens_comb_masked)
        return F.log_softmax(decoded)  # logsoftmax--->NLLLoss
        '''
        d_score = self.get_scores_d_all(vids_long,hiddens_comb_masked)
        all_comb = torch.cat((hiddens_long, hiddens_short, embeddings_t,embeddings_u,d_score), 1)
        decoded = self.decode_all(all_comb)
        
        return F.log_softmax(decoded) #logsoftmax--->NLLLoss

    def get_scores_d_all(self,vids_long,hiddens_comb_masked):
        for vid in vids_long:
            distance_vid_score = self.get_scores_d(vid)
            d_score = 0
        return d_score

    def get_scores_d(self,vid):
        vid_distance =
        distance_vid_score =
        return 0


'''