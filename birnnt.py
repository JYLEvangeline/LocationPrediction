import torch
import torch.nn as nn
from birnn import BiRNN
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
# use_cuda = False

class BiRNNT(BiRNN):
    def __init__(self, v_size, t_size, emb_dim_v, emb_dim_t, hidden_dim):
        super(BiRNNT, self).__init__(v_size, emb_dim_v, hidden_dim)
        self.t_size = t_size
        self.emb_dim_t = emb_dim_t
        self.embedder_t = nn.Embedding(t_size, self.emb_dim_t, padding_idx=0)
        self.decoder_dim = self.hidden_dim * 2 + self.emb_dim_t
        self.decoder = nn.Linear(self.decoder_dim, self.v_size)

    def get_embedding_t(self, tids_next, len_long, mask_long_valid):
        tids_next_strip = tids_next.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        embedding_t = self.embedder_t(tids_next_strip).view(-1, self.emb_dim_t)
        embedding_t_valid = embedding_t.masked_select(mask_long_valid.view(-1, 1).expand_as(embedding_t)).view(-1, self.emb_dim_t)
        return embedding_t_valid

    def forward(self, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate):
        mask_long_valid = mask_long.index_select(1, Variable(torch.LongTensor(range(torch.max(len_long).data[0]))))
        mask_optim_valid = (mask_optim if len(mask_evaluate) == 0 else mask_evaluate).index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long).data[0])))).masked_select(mask_long_valid)
        embeddings_t = self.get_embedding_t(tids_next, len_long, mask_long_valid)
        hiddens_long = self.get_hiddens_long(vids_long, len_long, mask_long_valid)
        hiddens_short = self.get_hiddens_short(vids_short_al, len_short_al,short_cnt)
        hiddens_comb = torch.cat((hiddens_long, hiddens_short, embeddings_t), 1)
        mask_optim_expanded = mask_optim_valid.view(-1, 1).expand_as(hiddens_comb)
        hiddens_comb_masked = hiddens_comb.masked_select(mask_optim_expanded).view(-1, self.decoder_dim)
        decoded = self.decoder(hiddens_comb_masked)
        return F.log_softmax(decoded) #logsoftmax--->NLLLoss