import numpy as np
def kfold(n:int,k:int=5,seed:int=520):
    """
    Generate k-fold train/test splits.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of folds.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    row = np.arange(n)
    row_ = row.copy()
    choices = []
    bpoint = np.linspace(0,n,k+1,dtype=int)
    nums = bpoint[1:]-bpoint[:-1]
    test_train_list = []
    for _ in range(k):
        row_ = row_[~np.isin(row_,choices)]
        testrow = np.random.choice(row_,nums[_],replace=False) # 不放回抽样
        trainrow = row[~np.isin(row,testrow)]
        choices.extend(testrow)
        test_train_list.append((testrow,trainrow))
    return test_train_list
