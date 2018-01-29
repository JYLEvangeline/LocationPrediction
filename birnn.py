import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np
use_cuda = torch.cuda.is_available()
# use_cuda = False

class BiRNN(nn.Module):
    def __init__(self, v_size, emb_dim_v, hidden_dim):
        super(BiRNN, self).__init__()
        self.v_size = v_size
        self.emb_dim_v = emb_dim_v
        self.hidden_dim = hidden_dim
        self.rnn_short = nn.RNN(self.emb_dim_v, self.hidden_dim, batch_first=True)
        self.rnn_long = nn.GRU(self.emb_dim_v, self.hidden_dim, batch_first=True)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v, padding_idx=0)
        self.decoder = nn.Linear(self.hidden_dim * 2, self.v_size)

    def get_hiddens_short(self, vids_short_al, len_short_al, short_cnt):
        session_cnt_total = torch.sum(short_cnt, dim=0).data[0]
        max_session_length_a_user = torch.max(len_short_al).data[0]
        user_cnt = vids_short_al.size(0)
        idx_al = []
        idx = 0
        for u in xrange(user_cnt):
            idx_u = []
            for session_cnt in xrange(short_cnt.data[u, 0]):
                for i in xrange(max_session_length_a_user):
                    if i < len_short_al.data[u, session_cnt]:
                        idx_u.append(idx)
                    idx += 1
            for i in idx_u:
                idx_al.append(i)
        seq_size = vids_short_al.size(2)
        vids_short_al_linear = vids_short_al.view(-1, seq_size)
        len_short_al_linear = len_short_al.view(-1)
        len_short_sorted, idx_sorted = len_short_al_linear.topk(k=session_cnt_total, largest=True, sorted=True)
        vids_short_al_valid = vids_short_al_linear.index_select(0, idx_sorted)
        vids_embeddings_sorted = self.embedder_v(vids_short_al_valid)
        vids_embeddings_sorted_packed = pack_padded_sequence(vids_embeddings_sorted, len_short_sorted.data.numpy(),
                                                             batch_first=True)#pack_padded_sequence
        hidden_short = self.init_hidden(session_cnt_total)
        hiddens_short, hidden_short = self.rnn_short(vids_embeddings_sorted_packed, hidden_short)
        unpack_hiddens, unpacked_len = pad_packed_sequence(hiddens_short, batch_first=True)
        _, idx_original = idx_sorted.sort(0, descending=False)
        index_original_idx = idx_original.view(-1, 1, 1).expand_as(unpack_hiddens)
        hiddens_unsorted = unpack_hiddens.gather(0, index_original_idx.long())
        hiddens_unsorted_linear = hiddens_unsorted.view(-1, self.hidden_dim)
        hiddens_unsorted_linear_valid = hiddens_unsorted_linear.index_select(0, Variable(torch.LongTensor(idx_al)))
        return hiddens_unsorted_linear_valid

    def get_hiddens_long(self, vids_long, len_long, mask_long_valid):
        len_long_sorted, idx_sorted = len_long.sort(0, descending=True)
        index_sorted_idx = idx_sorted.view(-1, 1).expand_as(vids_long) #21,26,24->[21 21..21;26 26..26;24 24..24]
        vids_long_sorted = vids_long.gather(0, index_sorted_idx.long()) # sorted according to lenth why?
        try:
            vids_embeddings_sorted = self.embedder_v(vids_long_sorted)
        except:
            for i in xrange(vids_long_sorted.size(0)):
                for j in xrange(vids_long_sorted.size(1)):
                    if vids_long_sorted.data[i, j] != 0:
                        print vids_long_sorted.data[i, j], vids_long_sorted.data[i, j] < self.v_size
            raw_input()
        #vids_embeddings_sorted_packed = pack_padded_sequence(vids_embeddings_sorted, len_long_sorted.data.numpy(), batch_first=True)
        vids_embeddings_sorted_packed = pack_padded_sequence(vids_embeddings_sorted, np.reshape(len_long_sorted.data.numpy(),(len_long_sorted.shape[0])), batch_first=True)
        hidden_long = self.init_hidden(vids_long.size(0))
        hiddens_long, hidden_long = self.rnn_long(vids_embeddings_sorted_packed, hidden_long)
        unpack_hiddens, unpacked_len = pad_packed_sequence(hiddens_long, batch_first=True)
        _, idx_original = idx_sorted.sort(0, descending=False)
        index_original_idx = idx_original.view(-1, 1, 1).expand_as(unpack_hiddens)
        output = unpack_hiddens.gather(0, index_original_idx.long())
        output_linear = output.view(-1, self.hidden_dim)
        mask_long_valid_expand = mask_long_valid.view(-1, 1).expand_as(output_linear)
        return output_linear.masked_select(mask_long_valid_expand).view(-1, self.hidden_dim)

    def forward(self, vids_long, len_long, vids_short_al, len_short_al, tids_next, short_cnt, mask_long, mask_optim, mask_evaluate):
        mask_long_valid = mask_long.index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long).data[0])))) # is mask_long? get which is valid here
        mask_optim_valid = (mask_optim if len(mask_evaluate) == 0 else mask_evaluate).index_select(1, Variable(torch.LongTensor(xrange(torch.max(len_long).data[0])))).masked_select(mask_long_valid)
        #mask_optim_valid = mask_optim.masked_select(mask_long_valid)
        # mask_optim is to delete invalid 0 and only keep 1
        hiddens_long = self.get_hiddens_long(vids_long, len_long, mask_long_valid)
        hiddens_short = self.get_hiddens_short(vids_short_al, len_short_al,short_cnt)
        hiddens_comb = torch.cat((hiddens_long, hiddens_short), 1)
        mask_optim_expanded = mask_optim_valid.view(-1, 1).expand_as(hiddens_comb)
        hiddens_comb_masked = hiddens_comb.masked_select(mask_optim_expanded).view(-1, self.hidden_dim * 2)#mask_select keep 1 delete 0 [3,5,2] [1,0,1]=>[3,2]
        decoded = self.decoder(hiddens_comb_masked)
        return F.log_softmax(decoded)

    def init_hidden(self, batch_size=1):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if use_cuda:
            hidden = hidden.cuda()
        return hidden