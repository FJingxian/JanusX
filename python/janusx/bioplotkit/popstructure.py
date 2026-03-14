import matplotlib.pyplot as plt
import numpy as np

thr = 0.7
palette = "tab10"
qmatrix = np.genfromtxt("test/mouse_hs1940.3.Q")
K = qmatrix.shape[1]
colors = plt.get_cmap(palette).colors
colors = {i: colors[i % len(colors)] for i in range(K)}
pop = qmatrix.argmax(axis=1)
pop[qmatrix.max(axis=1)<thr] = -1

fig = plt.figure(figsize=(10,3),dpi=300)
start = 0
for k in range(K):
    idx = np.flatnonzero(pop == k)
    if idx.size == 0:
        continue
    order = np.argsort(qmatrix[idx, k])
    submatrix = qmatrix[idx[order]]
    plt.bar(x=idx[order].astype(str),height=submatrix[:,k],bottom=0,color=colors[k],width=1)
    bot = [k]
    for i in range(K):
        if i!=k:
            plt.bar(x=idx[order].astype(str),height=submatrix[:,i],bottom=submatrix[:,bot].sum(axis=1),color=colors[i],width=1)
            bot.append(i)
    plt.vlines(start,0,1,color='black',linewidth=1)
    start += len(idx)
plt.vlines(start,0,1,color='black')
plt.xlim(0,start)
plt.ylim(0,1)
plt.xticks([])
plt.savefig('admix.png')