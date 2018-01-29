import torch
import numpy as np


input = torch.from_numpy(np.array([[1,2,3,4],[10,6,7,8],[9,10,11,12]]))
#print input[3:4,1:2]

length = [4,4,2,1] # lengths array has to be sorted in decreasing order
result,batch_sizes = torch.nn.utils.rnn.pack_padded_sequence(input,lengths=length,batch_first=False)
#print input[3:4,:1]
print result
print batch_sizes

Variable containing:
    1     1     1  ...      0     0     0
    1     1     1  ...      0     0     0
    1     1     1  ...      0     0     0
       ...          ⋱          ...
    1     1     1  ...      0     0     0
    1     1     1  ...      0     0     0
    1     1     1  ...      0     0     0
[torch.ByteTensor of size 50x384]
1
1
0
1
0
1
1
1
0

1
0
1
0
1
0
1
0
1

1
1
0
1
0
1
1
1
0

1
0
1
0
1
0
1