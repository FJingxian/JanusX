import pandas as pd
import numpy as np
def readanno(annofile:str,descItem:str='description'):
    '''After treating: anno 0-chr,1-start,2-end,3-geneID,4-description1,5-description2'''
    suffix = annofile.replace('.gz','').split('.')[-1]
    if suffix == 'bed':
        anno = pd.read_csv(annofile,sep='\t',header=None,).fillna('NA')
        if anno.shape[1]<=4:
            anno[4] = ['NA' for _ in anno.index]
        if anno.shape[1]<=5:
            anno[5] = ['NA' for _ in anno.index]
    elif suffix == 'gff' or suffix == 'gff3':
        anno = pd.read_csv(annofile,sep='\t',header=None,comment='#',low_memory=False,usecols=[0,2,3,4,8])
        anno = anno[(anno[2]=='gene')&(anno[0].astype(str).isin(np.arange(1,50).astype(str)))]
        del anno[2]
        anno[0] = anno[0].astype(int)
        anno = anno.sort_values([0,3])
        anno.columns = range(anno.shape[1])
        anno[4] = anno[3].str.extract(fr'{descItem}=(.*?);').fillna('NA')
        anno[3] = anno[3].str.extract(r'ID=(.*?);').fillna('NA')
        anno[5] = ['NA' for _ in anno.index]
    return anno