import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Util import IndexLinear
from Loss import MyNLLLoss
from Loss import NSNLLLoss
import string
import random
import pickle
import torch.optim as optim
torch.manual_seed(0)

class SERM(nn.Module):
    def __init__(self, u_size, v_size, t_size, w_size, emb_dim_all=50, emb_dim_v=50, emb_dim_t=50, emb_dim_w=50,
                 nb_cnt=100, sampling_list=None, glove_path=None, mod=1):
        super(SERM, self).__init__()
        self.v_size = v_size
        self.emb_dim_all = emb_dim_all
        self.emb_dim_v = emb_dim_v
        self.emb_dim_t = emb_dim_t
        self.emb_dim_w = emb_dim_w
        self.mod = mod
        self.nb_cnt = min((nb_cnt, v_size))
        self.sampling_list = sampling_list
        self.embedder_v = nn.Embedding(v_size, emb_dim_v)
        self.embedder_t = nn.Embedding(t_size, emb_dim_t)
        self.embedder_u = nn.Embedding(u_size, v_size)
        if mod == 0:
            self.embedder_w = nn.Embedding(w_size, emb_dim_w)
            self.gru_cell = nn.GRUCell(emb_dim_v + emb_dim_t + emb_dim_w, emb_dim_all)
            # read glove pre-trained embeddings
            if glove_path is not None:
                glove_file = open(glove_path, 'rt', -1)
                for line in glove_file:
                    wid = string.atoi(line[0: line.index('\t')])
                    probs = line[line.index('\t') + 1: -1].split(' ')
                    for i in range(emb_dim_w):
                        prob = string.atof(probs[i])
                        self.embedder_w.weight.data[wid, i] = prob
                glove_file.close()
        else:
            self.gru_cell = nn.GRUCell(emb_dim_v + emb_dim_t, emb_dim_all)
        self.decoder = IndexLinear(emb_dim_all, v_size)

    def forward(self, records_u, is_train):
        vids_true = []
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=1)
        emb_u = self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        predicted_scores = Variable(
            torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else Variable(
            torch.zeros(records_u.get_predicting_records_cnt(mod=2), self.v_size))
        idx = 0
        for rid, record in enumerate(records_al[0: len(records_al) - 1]):
            if record.is_first:
                hidden = self.init_hidden()
            if self.mod == 0:
                if len(record.wids) > 0:
                    emb_w = torch.mean(self.embedder_w(Variable(torch.LongTensor([record.wids])).view(1, -1)), 1)
                else:
                    emb_w = Variable(torch.zeros(1, 1, self.emb_dim_w))
            emb_v = self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1))
            emb_t = self.embedder_t(Variable(torch.LongTensor([record.tid])).view(1, -1))
            if self.mod == 0:
                emb_cat = torch.cat((emb_t.view(-1, self.emb_dim_t), emb_w.view(-1, self.emb_dim_w), emb_v.view(-1, self.emb_dim_v)), 1)
            else:
                emb_cat = torch.cat((emb_t.view(-1, self.emb_dim_t), emb_v.view(-1, self.emb_dim_v)), 1)
            hidden = self.gru_cell(emb_cat, hidden)
            if is_train:
                if record.is_last:
                    continue
                vids_true.append(record.vid_next)
                vid_candidates = self.get_vid_candidates(record.vid_next)
                output = self.decoder(hidden, vid_candidates.view(1, -1)) + torch.index_select(emb_u.view(-1), 0, vid_candidates)
                scores = F.softmax(output)
                predicted_scores[idx] = scores
            else:
                if record.is_last:
                    continue
                vids_true.append(record.vid_next)
                output = self.decoder(hidden) + emb_u.view(-1)
                scores = F.softmax(output)
                predicted_scores[idx] = scores
            idx += 1
        return predicted_scores, vids_true

    def get_vid_candidates(self, vid):
        reject = set()
        reject.add(vid)
        vid_candidates = [vid]
        while len(reject) <= self.nb_cnt:
            vid_candidate = self.sampling_list[random.randint(0, len(self.sampling_list) - 1)]
            if vid_candidate not in reject:
                reject.add(vid_candidate)
                vid_candidates.append(vid_candidate)
        return Variable(torch.LongTensor(vid_candidates))

    def init_hidden(self):
        return Variable(torch.zeros(1, self.emb_dim_all))


def train(root_path, emb_dim_all=50, emb_dim_v=50, emb_dim_t=50, emb_dim_w=50, nb_cnt=100, n_iter=500, iter_start=0, mod=0):
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    model = SERM(dl.nu, dl.nv, dl.nt, dl.nw, emb_dim_all, emb_dim_v, emb_dim_t, emb_dim_w,
                 nb_cnt=nb_cnt, sampling_list=dl.sampling_list, glove_path=root_path + 'glove.txt', mod=mod)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_serm_' + str(mod) + '_' + str(iter_start) + '.md'))
    optimizer = optim.Adam(model.parameters())
    criterion = NSNLLLoss()
    uids = dl.uid_records.keys()
    for iter in range(iter_start + 1, n_iter + 1):
        print_loss_total = 0
        random.shuffle(uids)
        for idx, uid in enumerate(uids):
            records_u = dl.uid_records[uid]
            optimizer.zero_grad()
            predicted_probs, _ = model(records_u, True)
            loss = criterion(predicted_probs)
            loss.backward()
            print_loss_total += loss.data[0]
            optimizer.step()
            if idx % 50 == 0:
                print '\t%d\t%f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 5 == 0:
            torch.save(model.state_dict(), root_path + 'model_serm_' + str(mod) + '_' + str(iter) + '.md')

def test(root_path, emb_dim_all=50, emb_dim_v=50, emb_dim_t=50, emb_dim_w=50, nb_cnt=500, iter_start=0, mod=0):
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    model = SERM(dl.nu, dl.nv, dl.nt, dl.nw, emb_dim_all, emb_dim_v, emb_dim_t, emb_dim_w,
                 nb_cnt=nb_cnt, sampling_list=dl.sampling_list, glove_path=root_path + 'glove.txt', mod=mod)
    model.load_state_dict(torch.load(root_path + 'model_serm_' + str(mod) + '_' + str(iter_start) + '.md'))
    hits = np.zeros(3)
    cnt = 0
    for uid, records_u in dl.uid_records.items():
        predicted_probs, vids_true = model(records_u, False)
        for idx in range(0, len(vids_true)):
            probs_sorted, vid_sorted = torch.sort(predicted_probs[idx].view(-1), 0, descending=True)
            cnt += 1
            for j in range(10):
                if vids_true[idx] == vid_sorted.data[j]:
                    if j == 0:
                        hits[0] += 1
                    if j < 5:
                        hits[1] += 1
                    if j < 10:
                        hits[2] += 1
        if (uid + 1) % 100 == 0:
            print (uid + 1), hits / cnt
    hits /= cnt
    print hits, cnt
