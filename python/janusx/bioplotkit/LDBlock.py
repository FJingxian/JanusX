import numpy as np
import matplotlib.patches as mpathes
import itertools
import matplotlib.pyplot as plt

from janusx.gfreader.gfreader import load_genotype_chunks

def LDblock(C:np.ndarray,ax:plt.Axes,vmin:float=0,vmax:float=1,cmap:str='Greys'):
    '''
    ax.set_xlim(0,n), 0.5 ~ n-0.5 n points, n=C.shape[0]=C.shape[1]
    '''
    cbar_rate = 0.05
    mask = np.triu(np.ones_like(C, dtype=bool), k=0) # 不包括对角线的上三角
    C[mask] = np.nan
    n = C.shape[0]
    # 创建旋转/缩放矩阵 (45度旋转伴随缩放)
    t = np.array([[1, 0.5], [-1, 0.5]])
    # 创建坐标矩阵并变换它
    A = np.dot(np.array([(i[1], i[0]) for i in itertools.product(range(n,-1,-1), range(0,n+1,1))]), t)
    # 绘制
    ax.pcolormesh(A[:,1].reshape(n+1,n+1), A[:,0].reshape(n+1,n+1), np.flipud(C),cmap=cmap,vmin=vmin,vmax=vmax)
    rect = mpathes.Rectangle((n/4,-n-cbar_rate*n),n/2,cbar_rate*n,fill=False, edgecolor='black', linewidth=.5)
    ax.add_patch(rect)
    gradient = np.linspace(vmin, vmax, 10).reshape(1, -1)
    clabel = np.round(np.arange(vmin, vmax + 1e-9, 0.2), 2)
    if clabel.size == 0 or not np.isclose(clabel[-1], vmax):
        clabel = np.append(clabel, round(vmax, 2))
    cticks = n/4 + (clabel - vmin) / max(vmax - vmin, 1e-12) * (n/2)
    for i in range(len(clabel)):
        ax.text(cticks[i],-n,clabel[i],ha='center')
    ax.imshow(gradient, cmap=cmap, aspect='auto', extent=[n/4, 3*n/4, -n - cbar_rate * n, -n], origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-n-cbar_rate*n,0)
    ax.set_xlim(0.5,n-0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect(0.5, adjustable="box")  # 对应当前 t 的 x 压缩

    
if __name__ == '__main__':
    chunks = load_genotype_chunks('example/mouse_hs1940.vcf.gz')
    for chunk,site in chunks:
        pass
    cor = np.corrcoef(chunk[:1000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    LDblock(cor,ax)
    plt.show()
