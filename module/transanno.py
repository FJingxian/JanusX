import pandas as pd
import sys
from _readanno import readanno

filterfile = sys.argv[1]
idtransfile = sys.argv[2]
annofile = sys.argv[3]
try:
    annobroaden = int(sys.argv[4])
except:
    annobroaden = None
descItem = "description"
print('* Annotating...')
df_filter = pd.read_csv(filterfile,sep='\t').set_index(['#CHROM','POS'])
df_trans = pd.read_csv(idtransfile,sep='\t',header=None).set_index([0,1])
df_filter = df_filter.loc[df_filter.index.isin(df_trans.index)]
anno = readanno(annofile,descItem) # After treating: anno 0-chr,1-start,2-end,3-geneID,4-description1,5-description2
desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1])&(anno[2]>=x[1])], df_trans.loc[df_filter.index].values))
df_filter['trans_desc'] = list(map(lambda x:f'''{x.iloc[0,3]};{x.iloc[0,4]};{x.iloc[0,5]}''' if not x.empty else 'NA;NA;NA', desc))
if annobroaden is not None:
    desc = list(map(lambda x:anno.loc[(anno[0]==x[0])&(anno[1]<=x[1]+annobroaden*1_000)&(anno[2]>=x[1]-annobroaden*1_000)], df_trans.loc[df_filter.index].values))
    df_filter['trans_broaden'] = list(map(lambda x:f'''{'|'.join(x.iloc[:,3])};{'|'.join(x.iloc[:,4])};{'|'.join(x.iloc[:,5])}''' if not x.empty else 'NA;NA;NA', desc))
print(df_filter)
df_filter.to_csv(f'{filterfile}.trans',sep='\t')
print(f'Saved in {filterfile}.trans')
