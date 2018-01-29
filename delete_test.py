import numpy as np
import torch
from torch.autograd import Variable

class network(torch.nn.Module):
    def __init__(self,in_num,hidden_num,out_num):
        super(network,self).__init__()
        self.input_layer=torch.nn.Linear(in_num,hidden_num)
        self.sigmoid=torch.nn.Sigmoid()
        self.output_layer=torch.nn.Linear(hidden_num,out_num)
        self.softmax=torch.nn.LogSoftmax()
    def forward(self,input_x):
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.softmax(self.output_layer(h_1))
        return h_2

in_num=100
hidden_num=50
out_num=2
batch_n=8
input_data = Variable(torch.randn(batch_n,in_num))
target = np.zeros([batch_n],dtype=np.int64)
print input_data
print target
for idx,t in enumerate(target):
    if idx%2==0:
        target[idx]=1
    else:target[idx]=0

target = Variable(torch.from_numpy(target))
net=network(in_num,hidden_num,out_num)
loss_function=torch.nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
for i in range(100):
    out=net(input_data)
    #print out
    loss=loss_function(out,target)
    print ("loss is %f"%loss.data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()