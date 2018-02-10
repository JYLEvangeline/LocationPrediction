import numpy as np
a = [[1,2],[3,4],[5,6]]
a = np.array(a)
q = []
for i in range(len(a)):
    q.append([])
    for j in range(i+1,len(a)):
        q[i].append(np.sqrt(np.sum((a[i] - a[j]) ** 2)))
        #q[i][j] = np.sqrt(np.sum((a[i] - a[j]) ** 2))