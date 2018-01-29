import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from Loss import MyNLLLoss
#from Loss import NSNLLLoss
#import string
import random
import pickle
import torch.optim as optim
#from Loss import BPRLoss


class STRNN(nn.Module):
    def __init__(self, u_size, v_size, emb_dim=10, w=6, vid_coor=None, sampling_list=None, d_al=None, t_al=None):
        super(STRNN, self).__init__()
        self.emb_dim = emb_dim
        self.v_size = v_size
        self.vid_coor = vid_coor
        self.sampling_list = sampling_list
        self.w = w
        self.t_al = t_al
        self.d_al = d_al
        self.embedder_u = nn.Embedding(u_size, emb_dim)
        self.embedder_v = nn.Embedding(v_size, emb_dim)
        self.T = []
        self.S = []
        for i in range(len(t_al)):
            self.T.append(nn.Parameter(torch.randn(emb_dim, emb_dim)))
        for i in range(len(d_al)):
            self.S.append(nn.Parameter(torch.randn(emb_dim, emb_dim)))
        self.C = nn.Linear(emb_dim, emb_dim)

    def forward(self, records_u, is_train):
        emb_u = self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=2)
        hiddens = []
        predicted_scores = Variable(
            torch.zeros(records_u.get_predicting_records_cnt(mod=0), 1)) if is_train else Variable(
            torch.zeros(records_u.get_predicting_records_cnt(mod=2), self.v_size))
        vids_true = []
        idx = 0
        for rid, record in enumerate(records_al[0: len(records_al) - 1]):
            rid_w_start = rid
            while rid_w_start - 1 > 0 and (record.dt - records_al[rid_w_start - 1].dt).total_seconds() / 3600.0 < self.w:
                rid_w_start -= 1
            if rid != rid_w_start:
                input = self.C(hiddens[rid_w_start].view(1, -1))
            else:
                input = self.C(self.init_hidden().view(1, -1))
            for rrid in range(rid_w_start, rid):
                record_r = records_al[rrid]
                gap = (record.dt - record_r.dt).total_seconds()
                dis = np.sqrt((self.vid_coor[record.vid] - self.vid_coor[record_r.vid]).__pow__(2).sum())
                # print gap, dis
                t_l, t_u = self.find_bin_idx(gap, self.t_al)
                d_l, d_u = self.find_bin_idx(dis, self.d_al)
                T_weight = (self.T[t_l] * (self.t_al[t_u] - gap) + self.T[t_u] * (gap - self.t_al[t_l])) / \
                           (self.t_al[t_u] - self.t_al[t_l])
                S_weight = (self.S[d_l] * (self.d_al[d_u] - dis) + self.S[d_u] * (dis - self.d_al[d_l])) / \
                           (self.d_al[d_u] - self.d_al[d_l])
                emb_v_r = self.embedder_v(Variable(torch.LongTensor([record_r.vid])).view(1, -1)).view(1, -1)
                input += F.linear(emb_v_r, torch.mm(T_weight, S_weight))
            hidden = F.sigmoid(input)
            hiddens.append(hidden)
            if record.is_last:
                continue
            if is_train:
                vid_true = record.vid_next
                vids_true.append(vids_true)
                vid_noise = self.sampling_list[random.randint(0, len(self.sampling_list) - 1)]
                while vid_noise == vid_true:
                    vid_noise = self.sampling_list[random.randint(0, len(self.sampling_list) - 1)]
                emb_v_true = self.embedder_v(Variable(torch.LongTensor([vid_true])).view(1, -1)).view(-1)
                emb_v_noise = self.embedder_v(Variable(torch.LongTensor([vid_noise])).view(1, -1)).view(-1)
                score_diff = torch.sum((hidden + emb_u) * (emb_v_true - emb_v_noise))
                predicted_scores[idx] = score_diff
                idx += 1
            else:
                if rid >= records_u.test_idx:
                    vids_true.append(record.vid_next)
                    scores = Variable(torch.zeros(self.v_size))
                    for vid in range(0, self.v_size):
                        emb_v = self.embedder_v(Variable(torch.LongTensor([vid])).view(1, -1)).view(-1)
                        score = torch.sum((hidden + emb_u) * emb_v)
                        scores[vid] = score
                    predicted_scores[idx] = scores
                    idx += 1
        return predicted_scores, vids_true

    def find_bin_idx(self, val, al):
        idx = 0
        try:
            while al[idx] <= val:
                idx += 1
        except:
            print val
            print al
        return idx - 1, idx

    def init_hidden(self):
        return Variable(torch.zeros(1, self.emb_dim))

def get_dis_time_als(dl, bin_cnt, w):
    dis_min = 10000000
    dis_max = -1
    for uid, records_u in dl.uid_records.items():
        coor_first = None
        coor_al = []
        records_al = records_u.get_records(mod=2)
        for rid, record in enumerate(records_al[0: len(records_al) - 1]):
            coor = dl.vid_coor[record.vid]
            if len(coor_al) > 0:
                for coor_pre in coor_al:
                    dis = np.sqrt((coor - coor_pre).__pow__(2).sum())
                    dis_min = min([dis, dis_min])
                    dis_max = max([dis, dis_max])
            coor_al.append(coor)
    dis_max += dis_max/10000
    d_al = [dis_min]
    t_al = [0]
    int_dis = (dis_max - dis_min) / bin_cnt
    int_time = w * 3600.0 / bin_cnt
    for i in range(1, bin_cnt + 1):
        d_al.append(dis_min + int_dis * i)
        t_al.append(int_time * i)
    return d_al, t_al

def train(root_path, emb_dim=10, w=6, n_iter=500, iter_start=0):
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    d_al, t_al = get_dis_time_als(dl, 10, w)
    model = STRNN(dl.nu, dl.nv, emb_dim, w, vid_coor=dl.vid_coor, sampling_list=dl.sampling_list, d_al=d_al, t_al=t_al)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_strnn_' + str(iter_start) + '.md'))
    optimizer = optim.Adam(model.parameters())
    #criterion = BPRLoss()
    criterion = nn.NLLLoss()
    uids = dl.uid_records.keys()
    for iter in range(iter_start + 1, n_iter + 1):
        print_loss_total = 0
        random.shuffle(uids)
        for idx, uid in enumerate(uids):
            records_u = dl.uid_records[uid]
            optimizer.zero_grad()
            predicted_scores, _ = model(records_u, True)
            loss = criterion(predicted_scores)
            loss.backward(retain_variables=True)
            print_loss_total += loss.data[0]
            optimizer.step()
            if idx % 50 == 0:
                print '\t%d\t%f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 5 == 0:
            torch.save(model.state_dict(), root_path + 'model_strnn_' + str(iter) + '.md')

def test(root_path, emb_dim=10, w=6, iter_start=0):
    dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
    d_al, t_al = get_dis_time_als(dl, 10, w)
    model = STRNN(dl.nu, dl.nv, emb_dim, w, vid_coor=dl.vid_coor, sampling_list=dl.sampling_list, d_al=d_al, t_al=t_al)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_strnn_' + str(iter_start) + '.md'))
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
        if (uid + 1) % 50 == 0:
            print (uid + 1), hits / cnt
    hits /= cnt
    print hits, cnt
