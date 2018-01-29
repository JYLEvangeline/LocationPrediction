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

class JNTM(nn.Module):
    def __init__(self, u_size, v_size, emb_dim=50, nb_cnt=100, sampling_list=None, mod=0):
        super(JNTM, self).__init__()
        self.emb_dim = emb_dim
        self.u_size = u_size
        self.v_size = v_size
        self.nb_cnt = nb_cnt
        self.sampling_list = sampling_list
        self.mod = mod
        self.rnn_cell = nn.RNNCell(emb_dim, emb_dim)
        self.gru_cell = nn.GRUCell(emb_dim, emb_dim)
        self.embedder_u = nn.Embedding(u_size, emb_dim)
        self.embedder_v = nn.Embedding(v_size, emb_dim)
        if mod == 0:
            self.decoder = IndexLinear(emb_dim * 3, v_size)
        else:
            self.decoder = IndexLinear(emb_dim * 2, v_size)

    def forward(self, records_u, is_train):
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=2)
        emb_u = self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        if self.mod == 0 or self.mod == 1 or self.mod == 3 or self.mod == 4:
            hidden_long = self.init_hidden()
        predicted_scores = Variable(torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else Variable(
                torch.zeros(records_u.get_predicting_records_cnt(mod=2), self.v_size))
        vids_true = []
        idx = 0
        session_start_rid = 0
        for rid, record in enumerate(records_al[0: len(records_al) - 1]):
            if record.is_first and (self.mod == 0 or self.mod == 2 or self.mod == 3):
                hidden_short = self.init_hidden()
                session_start_rid = rid
            emb_v = self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)
            if self.mod == 0 or self.mod == 1 or self.mod == 3:
                hidden_long = self.gru_cell(emb_v, hidden_long)
            if self.mod == 0 or self.mod == 2 or self.mod == 3:
                hidden_short = self.rnn_cell(emb_v, hidden_short)
            if self.mod == 0:
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), hidden_long.view(-1, self.emb_dim),
                                hidden_short.view(-1, self.emb_dim)), 1)
            elif self.mod == 1:
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), hidden_long.view(-1, self.emb_dim)), 1)
            elif self.mod == 2:
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), hidden_short.view(-1, self.emb_dim)), 1)
            elif self.mod == 3:
                weight = np.tanh(0.333 * (rid - session_start_rid))
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), (hidden_long.view(-1, self.emb_dim) * (1 - weight) +
                                    hidden_short.view(-1, self.emb_dim) * weight)), 1)
            if record.is_last:
                continue
            if is_train:
                vids_true.append(record.vid_next)
                vid_candidates = self.get_vid_candidates(record.vid_next)
                output = self.decoder(hidden, vid_candidates.view(1, -1))
                predicted_scores[idx] = F.softmax(output)
                idx += 1
            else:
                if rid >= records_u.test_idx:
                    vids_true.append(record.vid_next)
                    output = self.decoder(hidden)
                    predicted_scores[idx] = F.softmax(output)
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
        return Variable(torch.zeros(1, self.emb_dim))

def train(root_path, emb_dim=50, nb_cnt=100, n_iter=500, iter_start=0, mod=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    model = JNTM(dl.nu, dl.nv, emb_dim, nb_cnt=nb_cnt, sampling_list=dl.sampling_list, mod=mod)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_jntm_' + str(mod) + '_' +str(iter_start) + '.md'))
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
            torch.save(model.state_dict(), root_path + 'model_jntm_'+ str(mod) + '_' + str(iter) + '.md')


def test(root_path, emb_dim=50, nb_cnt=100, iter_start=0, mod=0):
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    model = JNTM(dl.nu, dl.nv, emb_dim, nb_cnt=nb_cnt, sampling_list=dl.sampling_list, mod=mod)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_jntm_'+ str(mod) + '_' + str(iter_start) + '.md'))
    hits = np.zeros(3)
    cnt = 0
    for uid, records_u in dl.uid_records.items():
        predicted_probs, vid_true = model(records_u, False)
        for idx in range(0, len(vid_true)):
            probs_sorted, vid_sorted = torch.sort(predicted_probs[idx].view(-1), 0, descending=True)
            cnt += 1
            for j in range(10):
                if vid_true[idx] == vid_sorted.data[j]:
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