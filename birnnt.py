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
     
     def get_scores_d_all(self, records_u, idx_cur, vid_candidates, feature_al, is_train):  #id: current record id, want to predict record[id].vid_next
        feature_next = feature_al[idx_cur + 1].view(1, -1)
        coor_cur = self.vid_coor_nor[records_u.records[idx_cur].vid]
        records_al = records_u.records[0:records_u.test_idx if is_train else idx_cur + 1]
        atten_scores = Variable(torch.zeros(len(records_al)))
        for idx_r, record in enumerate(records_al):
            if idx_r == idx_cur + 1:
                atten_scores.data[idx_r] = float('-inf')
                continue
            coor_r = self.vid_coor_nor[records_u.records[idx_r].vid]
            dist = np.sqrt(np.sum((coor_cur - coor_r) ** 2))
            feature_r = feature_al[idx_r].view(-1, 1)
            feature_part = torch.mm(torch.mm(feature_next, self.att_M), feature_r)
            dist_part = Variable(torch.FloatTensor([dist]))
            score = self.att_merger(torch.cat((feature_part, dist_part), 0).view(1, -1))
            atten_scores[idx_r] = score
        atten_scores = F.softmax(atten_scores)

        scores_d = Variable(torch.zeros(1, len(vid_candidates)))
        for idx, vid_candidate in enumerate(vid_candidates):
            score_sum = Variable(torch.zeros([1]))
            for idx_r in xrange(len(records_al)):
                if idx_r == idx_cur + 1:
                    continue
                score = self.get_d_score(records_al[idx_r].vid, vid_candidate)
                score_sum += atten_scores[idx_r] * score
            scores_d[0, idx] = score_sum
        return scores_d
