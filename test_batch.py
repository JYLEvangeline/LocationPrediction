from torch.utils.data import Dataset, DataLoader
class Test(Dataset):
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

test = Test([[1,2,3],[4,5],[10,9,1]],['a','b','c'],[11,2,5])
print test
qaq = DataLoader(test, batch_size=3, shuffle=True, num_workers=1)
print qaq