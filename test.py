import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

distance_vids_score = np.array([[1,2,3,4],[8.3,5,6,0]])
a = np.full((len(distance_vids_score), len(distance_vids_score[0])), 0.00001)
b = Variable(torch.t(torch.LongTensor(distance_vids_score)))
q = (b==min(b)).nonzero()
print(b)
b.data[q.data[0,0],q.data[0,1]] = torch.LongTensor(b.max())


'''
m = distance_vids_score.min()
if m<0:
    b = np.full((len(distance_vids_score), len(distance_vids_score[0])), -m)
    distance_vids_score = (distance_vids_score+b)*a
    distance_vids_score = Variable(torch.LongTensor(distance_vids_score))
print(distance_vids_score)
'''
# v = Variable(torch.LongTensor(range(10)))
# print v
# idx = Variable(torch.LongTensor([0, 2, 1]))
# print v.index_select(0, idx)
# batch_size = 4
# max_length = 3
# hidden_size = 2
# n_layers =1
# num_input_features = 1
#
# input_tensor = torch.zeros(batch_size,max_length,num_input_features)
# input_tensor[0] = torch.FloatTensor([1,2,3])
# input_tensor[1] = torch.FloatTensor([4,5,0])
# input_tensor[2] = torch.FloatTensor([6,7,0])
# input_tensor[3] = torch.FloatTensor([8,0,0])
# batch_in = Variable(input_tensor)
# seq_lengths = [3,2,2,1]
# pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
# print (pack)
# unpack = torch.nn.utils.rnn.pad_packed_sequence(pack, seq_lengths)
# print unpack