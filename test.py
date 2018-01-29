import torch
import torch.nn as nn
from torch.autograd import Variable

v = Variable(torch.zeros(0))
print len(v)
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