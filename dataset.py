import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CheckinData(Dataset):
    def __init__(self, data_path, id_offset=0):
        self.uid_vids_long = []
        self.uid_vids_short_al = []
        self.uid_tids = []
        self.uid_vids_next = []
        self.uid_tids_next = []
        self.uid_test_idx = []

        f_data = open(data_path, 'r')
        lines = f_data.readlines()
        f_data.close()
        i = 0
        while i < len(lines):
            # read userid, #session
            uid, cnt, test_idx = map(int, lines[i].split(','))
            self.uid_test_idx.append(test_idx)
            i += 1
            # read whole trajectory
            self.uid_vids_long.append(np.fromstring(lines[i], dtype=np.int32, sep=',') + id_offset)
            i += 1
            # read sessions
            vids_short_al = []
            for j in xrange(cnt):
                vids_short_al.append(np.fromstring(lines[i + j], dtype=np.int32, sep=',') + id_offset)
            self.uid_vids_short_al.append(vids_short_al)
            i += cnt
            # self.uid_tids.append(map(int, lines[i].split(',')))
            self.uid_tids.append(np.fromstring(lines[i], dtype=np.int32, sep=',') + id_offset)
            i += 1
            self.uid_vids_next.append(np.fromstring(lines[i], dtype=np.int32, sep=',') + id_offset)
            i += 1
            self.uid_tids_next.append(np.fromstring(lines[i], dtype=np.int32, sep=',') + id_offset)
            i += 1
        self.max_long_len, self.max_short_len, self.max_session_len = self.get_max_len()

    def get_max_len(self):
        max_long_len = 0
        max_short_len = 0
        max_session_len = 0
        for seq in self.uid_vids_long:
            max_long_len = max((max_long_len, len(seq)))
        for vids_short_al in self.uid_vids_short_al:
            max_session_len = max(max_session_len, len(vids_short_al))
            for vids_short in vids_short_al:
                max_short_len = max(max_short_len, len(vids_short))
        return max_long_len, max_short_len, max_session_len

    def __len__(self):
        return len(self.uid_vids_long)

    def __getitem__(self, uid):
        vids_long = np.zeros(self.max_long_len, dtype=np.int)
        tids = np.zeros(self.max_long_len, dtype=np.int)
        vids_next = np.zeros(self.max_long_len, dtype=np.int)
        tids_next = np.zeros(self.max_long_len, dtype=np.int)
        uids_next = np.zeros(self.max_long_len, dtype=np.int)
        vids_short_al = np.zeros((self.max_session_len, self.max_short_len), dtype=np.int)
        mask_long = np.zeros(self.max_long_len, dtype=np.int)
        mask_test = np.zeros(self.max_long_len, dtype=np.int)
        len_long = len(self.uid_vids_long[uid])
        len_short_al = np.zeros(self.max_session_len, dtype=np.int)
        # short_len_cur = np.zeros(self.max_session_len, dtype=np.int)
        # idx = 0
        # for vids_short_al in self.uid_vids_short_al[uid]:
        #     for i in xrange(len(vids_short_al)):
        #         short_len_cur[idx] = i
        #         idx += 1
        # print self.uid_vids_short_al[uid]
        # print short_len_cur
        for i in xrange(len(self.uid_vids_short_al[uid])):
            len_short_al[i] = len(self.uid_vids_short_al[uid][i])
        for i in xrange(len_long):
            vids_long[i] = self.uid_vids_long[uid][i]
            tids[i] = self.uid_tids[uid][i]
            vids_next[i] = self.uid_vids_next[uid][i]
            tids_next[i] = self.uid_tids_next[uid][i]
            uids_next[i] = uid
            mask_long[i] = 1
        for i in xrange(len(len_short_al)):
            for j in xrange(len_short_al[i]):
                vids_short_al[i][j] = self.uid_vids_short_al[uid][i][j]
        idx = 0
        for i in xrange(len(self.uid_vids_short_al[uid])):
            for j in xrange(len(self.uid_vids_short_al[uid][i])):
                if j != len(self.uid_vids_short_al[uid][i]) - 1:
                    mask_test[idx] = 1
                idx += 1
            '''
        return torch.from_numpy(vids_long), torch.from_numpy(vids_short_al), torch.from_numpy(tids), \
               torch.LongTensor([len_long]), torch.from_numpy(len_short_al), \
               torch.from_numpy(mask_long).byte(), torch.from_numpy(mask_test).byte(), \
               torch.from_numpy(vids_next), torch.from_numpy(tids_next), \
               torch.LongTensor([uid]), torch.LongTensor([self.uid_test_idx[uid]]), torch.LongTensor([len(self.uid_vids_short_al[uid])])
               '''
        return torch.from_numpy(vids_long), torch.from_numpy(vids_short_al), torch.from_numpy(tids), \
               torch.LongTensor([len_long]), torch.from_numpy(len_short_al), \
               torch.from_numpy(mask_long).byte(), torch.from_numpy(mask_test).byte(), \
               torch.from_numpy(vids_next), torch.from_numpy(tids_next), \
               torch.from_numpy(uids_next), torch.LongTensor([self.uid_test_idx[uid]]), torch.LongTensor(
            [len(self.uid_vids_short_al[uid])])

class Vocabulary:
    def __init__(self, data_file, id_offset=0):
        self.id_offset = id_offset
        self.id_name = {}
        self.name_id = {}
        with open(data_file, 'r') as fin:
            for line in fin:
                al = line.strip().split(',')
                id = int(al[1]) + id_offset
                name = al[0]
                self.id_name[id] = name
                self.name_id[name] = id

    def size(self):
        return len(self.id_name) + self.id_offset

def read_distance(root_path):
    result = []
    with open(root_path, 'r') as f:
        for line in f.readlines():
            al = line.strip().split(',')
            al_float = map(eval,al)
            result.append(al_float)
    return result


class DataSet:
    def __init__(self, opt):
        u_vocab_file = opt['u_vocab_file']
        v_vocab_file = opt['v_vocab_file']
        t_vocab_file = opt['t_vocab_file']
        train_file = opt['train_data_file']
        test_file = opt['test_data_file']
        coor_file = opt['coor_nor_file']
        distance_file = opt['distance_file']
        batch_size = opt['batch_size']
        n_worker = opt['data_worker']
        id_offset = opt['id_offset']
        # print 'id_offset: ', id_offset
        self.u_vocab = Vocabulary(u_vocab_file, id_offset=0)
        self.v_vocab = Vocabulary(v_vocab_file, id_offset=id_offset)
        self.t_vocab = Vocabulary(t_vocab_file, id_offset=id_offset)
        train_data = CheckinData(train_file, id_offset=id_offset)
        test_data = CheckinData(test_file, id_offset=id_offset)
        vid_coor_nor = np.loadtxt(coor_file, delimiter=',', dtype=np.float64)
        #self.distance = read_distance(distance_file)
        self.distance = 0
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)

