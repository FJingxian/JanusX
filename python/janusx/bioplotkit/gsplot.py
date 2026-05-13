import matplotlib.pyplot as plt
import numpy as np


def mse(tparray:np.ndarray):
    se = tparray[:,0]-tparray[:,1]
    return se.T@se/se.shape[0]
def r2(tparray:np.ndarray):
    sse1 = tparray[:,0]-tparray[:,1]
    sse2 = tparray[:,0] - np.mean(tparray[:,0])
    return 1-sse1.T@sse1/(sse2.T@sse2)

def scatterh(
    ttest: np.ndarray,
    ttrain: np.ndarray = None,
    color_set: list = ['black', 'grey'],
    fig: plt.Figure = None,
    rasterized: bool | None = None,
):
    if not fig:
        fig = plt.figure(figsize=(5,4),dpi=300)
    layout = np.array([['A']*4+['.']+(['C']*4+['D'])*4]).reshape(5,5)
    axe = fig.subplot_mosaic(layout)
    ax1:plt.Axes = axe['A']
    ax2:plt.Axes = axe['C']
    ax3:plt.Axes = axe['D']
    if rasterized is None:
        n_train = int(ttrain.shape[0]) if (ttrain is not None) else 0
        rasterized_flag = True if n_train > 2000 else False  # reduce vector size on very large plots
    else:
        rasterized_flag = bool(rasterized)
    # scatterplot
    ax2.plot([np.min(ttest),np.max(ttest)],[np.min(ttest),np.max(ttest)],linestyle='--',color=color_set[0],alpha=.8,label='y = x (Ideal)')
    if ttrain is not None:
        ax2.scatter(ttrain[:,0],ttrain[:,1],color=color_set[1],alpha=.4,marker='+',label='Train data',rasterized=rasterized_flag)
    ax2.scatter(ttest[:,0],ttest[:,1],color=color_set[0],alpha=.8,marker='*',label='Test data',rasterized=rasterized_flag)
    ax2.set_xlabel('Observed Value')
    ax2.set_ylabel('Predicted Value')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='both')
    if ttrain is not None:
        stat_txt = (
            f'Train MSE: {mse(ttrain):.2f}\n'
            f'Test MSE: {mse(ttest):.2f}\n'
            f'Train R2: {r2(ttrain):.2f}\n'
            f'Test R2: {r2(ttest):.2f}'
        )
    else:
        stat_txt = f'Test MSE: {mse(ttest):.2f}\nTest R2: {r2(ttest):.2f}'
    ax2.text(
        -0.2,
        0.04,
        stat_txt,
        transform=ax2.transAxes,
        ha='right',
        va='bottom',
        color='gray',
        alpha=.8,
        multialignment='left',
    )
    # histplot
    if ttrain is not None:
        ax1.hist(ttrain[:,0],color=color_set[1],density=True,alpha=.4,bins=20)
    ax1.hist(ttest[:,0],color=color_set[0],density=True,alpha=.8,bins=20)
    if ttrain is not None:
        ax3.hist(ttrain[:,1],color=color_set[1],density=True,alpha=.4,orientation='horizontal',bins=20)
    ax3.hist(ttest[:,1],color=color_set[0],density=True,alpha=.8,orientation='horizontal',bins=20)
    # Scatter axes
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Hist axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    # adjust edge
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15,
        wspace=0.1, hspace=0.1)
if __name__ == '__main__':
    x = np.random.randn(100,1).reshape(-1,1)
    y = 2*x+np.random.randn(100,1).reshape(-1,1)
    x2 = np.random.randn(400,1).reshape(-1,1)
    y2 = 2*x2+np.random.randn(400,1).reshape(-1,1)
    scatterh(np.concatenate([y,2*x],axis=1),np.concatenate([y2,2*x2],axis=1))
