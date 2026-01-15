import time
from importlib.metadata import version, PackageNotFoundError
try:
    v = version("janusx")
except PackageNotFoundError:
    v = "0.0.0"
from itertools import takewhile,repeat
from .gfreader import load_genotype_chunks, inspect_genotype_file
import numpy as np
import pandas as pd
import gzip
import psutil
from tqdm import tqdm
import os
import psutil
process = psutil.Process()
def get_process_info():
    """Return current CPU utilization and resident memory usage."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024**3  # GB
    return cpu_percent, memory_mb

class GENOMETOOL:
    def __init__(self,genomePath:str):
        print(f"Adjusting reference/alternate alleles using {genomePath}...")
        chrom = []
        seqs = []
        with open(genomePath,'r') as f:
            for line in f:
                line = line.strip()
                if '>' in line:
                    if len(chrom) > 0:
                        seqs.append(seq)
                    chrom.append(line.split(' ')[0].replace('>',''))
                    seq = []
                else:
                    seq.append(line)
            seqs.append(seq)
        self.genome = dict(zip(chrom,seqs))
    def _readLoc(self,chr,loc):
        strperline = len(self.genome[f'{int(chr)}'][0])
        line = int(loc)//strperline
        strnum = int(loc)%strperline-1
        return self.genome[f'{chr}'][line][strnum]
    def refalt_adjust(self, ref_alt:pd.Series):
        ref_alt = ref_alt.astype(str)
        ref = pd.Series([self._readLoc(i,j) for i,j in ref_alt.index],index=ref_alt.index,name='REF')
        alt:pd.Series = ref_alt.iloc[:,0]*(ref_alt.iloc[:,0]!=ref)+ref_alt.iloc[:,1]*(ref_alt.iloc[:,1]!=ref)
        alt.name = 'ALT'
        ref_alt = pd.concat([ref,alt],axis=1)
        self.error = ref_alt.loc[alt.str.len()!=1,:].index
        ref_alt.loc[self.error,:] = pd.NA
        print(
            "Number of sites differing from the reference: "
            f"{len(self.error)} (ratio={round(len(self.error)/ref_alt.shape[0],3)})"
        )
        self.exchange_loc:bool = (ref_alt.iloc[:,0]!=ref)
        return ref_alt.astype('category')

def breader(prefix:str,ref_adjust:str=None,chunk_size=10_000,
            maf: float = 0,miss: float = 1, impute: bool=False) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    idv,m = inspect_genotype_file(prefix)
    chunks = load_genotype_chunks(prefix,chunk_size,maf,miss,impute)
    genotype = np.zeros(shape=(len(idv),m),dtype='int8')
    pbar = tqdm(total=m, desc="Loading bed",ascii=True)
    num = 0
    for chunk,_ in chunks:
        cksize = chunk.shape[0]
        genotype[:,num:num+cksize] = chunk.T
        num += cksize
        pbar.update(cksize)
    bim = pd.read_csv(f'{prefix}.bim',sep=r'\s+',header=None)
    genotype = pd.DataFrame(genotype,index=idv,).T
    genotype = pd.concat([bim[[0,3,4,5]],genotype],axis=1)
    genotype.columns = ['#CHROM','POS','A0','A1']+idv
    genotype = genotype.set_index(['#CHROM','POS'])
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype.dropna()

def vcfreader(vcfPath:str,chunk_size=50_000,ref_adjust:str=None,
            maf: float = 0,miss: float = 1, impute: bool=False) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    idv,m = inspect_genotype_file(vcfPath)
    chunks = load_genotype_chunks(vcfPath,chunk_size,maf,miss,impute)
    genotype = np.zeros(shape=(len(idv),m),dtype='int8')
    pbar = tqdm(total=m, desc="Loading bed",ascii=True)
    num = 0
    bim = []
    for chunk,site in chunks:
        cksize = chunk.shape[0]
        genotype[:,num:num+cksize] = chunk.T
        num += cksize
        pbar.update(cksize)
        bim.extend([[i.chrom,i.pos,i.ref_allele,i.alt_allele] for i in site])
    bim = pd.DataFrame(bim)
    genotype = pd.DataFrame(genotype,index=idv,).T
    genotype = pd.concat([bim,genotype],axis=1)
    genotype.columns = ['#CHROM','POS','A0','A1']+idv
    genotype = genotype.set_index(['#CHROM','POS'])
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype.dropna()

def hmpreader(hmp:str,sample_start:int=None,chr:str='chrom',ps:str='position',ref:str='ref',chunksize=10_000,ref_adjust:str=None):
    raws = pd.read_csv(hmp,sep='\t',chunksize=chunksize)
    _ = []
    for raw in raws:
        samples = raw.columns[sample_start:raw.shape[0]]
        genotype = raw[samples].fillna('XX')
        def filterindel(col:pd.Series):
            col[col.str.len()!=2] = 'XX'
            return col
        genotype = genotype.apply(filterindel,axis=0)
        ref_alt:pd.Series = genotype.sum(axis=1).apply(set).apply(''.join).str.replace('X','')
        biallele = ref_alt[ref_alt.str.len()==2]
        moallele = ref_alt[ref_alt.str.len()==1]
        moallele+=moallele
        mbiallele = pd.concat([biallele,moallele])
        mbiallele = mbiallele.str.split('',expand=True)[[1,2]]
        alt:pd.Series = mbiallele[1]*(mbiallele[1]!=raw.loc[mbiallele.index,ref])+mbiallele[2]*(mbiallele[2]!=raw.loc[mbiallele.index,ref])
        alt.loc[moallele.index] = raw.loc[moallele.index,ref]
        ref_alt:pd.DataFrame = pd.concat([raw[ref],alt],axis=1).dropna()
        ref_alt.columns = ['A0','A1']
        rr = ref_alt['A0']+ref_alt['A0']
        ra = ref_alt['A0']+ref_alt['A1']
        ar = ref_alt['A1']+ref_alt['A0']
        aa = ref_alt['A1']+ref_alt['A1']
        def hmp2genotype(col:pd.Series):
            return ((col==ra)|(col==ar)).astype('int8')+2*(col==aa).astype('int8')
        genotype = genotype.loc[ref_alt.index]
        xxmask = (genotype=='XX')
        genotype = genotype.apply(hmp2genotype,axis=0)
        genotype[xxmask] = -9
        genotype[genotype==3] = 0 # Fixed some bugs: 3 is combination of REF+REF
        chr_loc = raw.loc[genotype.index,[chr,ps]]
        chr_loc.columns = ['#CHROM','POS']
        genotype = pd.concat([chr_loc,ref_alt,genotype],axis=1)
        _.append(genotype.set_index(['#CHROM','POS']))
    genotype = pd.concat(_)
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def npyreader(prefix:str,ref_adjust:str=None):
    genotype = np.load(f'{prefix}.npz')
    samples = pd.read_csv(f'{prefix}.idv',sep='\t',header=None)[0].values
    ref_alt = pd.read_csv(f'{prefix}.snp',sep='\t',header=None,dtype={0:'category',1:'int32',2:'category',3:'category'})
    ref_alt.columns = ['#CHROM','POS','A0','A1']
    genotype:pd.DataFrame = pd.concat([ref_alt,pd.DataFrame(genotype['arr_0'],columns=samples)],axis=1).set_index(["#CHROM","POS"])
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def vcfinfo():
    import time
    alltime = time.localtime()
    vcf_info = f'''##fileformat=VCFv4.2
##fileDate={alltime.tm_year}{alltime.tm_mon}{alltime.tm_mday}
##source="JanusX-v{v}"
##INFO=<ID=PR,Number=0,Type=Flag,Description="Provisional reference allele, may not be based on real reference genome">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'''
    return vcf_info
    
def genotype2vcf(geno:pd.DataFrame,outPrefix:str=None,chunksize:int=10_000):
    import warnings
    warnings.filterwarnings('ignore')
    m,n = geno.shape
    vcf_head = 'ID QUAL FILTER INFO FORMAT'.split(' ')
    samples = geno.columns[2:]
    sample_duploc = samples.duplicated()
    dupsamples = ','.join(samples[sample_duploc])
    assert sample_duploc.sum()==0, f'Duplicated samples: {dupsamples}'
    with open(f'{outPrefix}.vcf','w') as f:
        f.writelines(vcfinfo())
    pbar = tqdm(total=m, desc="Saving as VCF",ascii=True)
    for i in range(0,m,chunksize):
        i_end = np.min([i+chunksize,m])
        g_chunk = np.full((i_end-i,n-2), './.', dtype=object)
        g_chunk[geno.iloc[i:i_end,2:]==0] = '0/0'
        g_chunk[geno.iloc[i:i_end,2:]==2] = '1/1'
        g_chunk[geno.iloc[i:i_end,2:]==1] = '0/1'
        info_chunk = geno.iloc[i:i_end,:2].reset_index()
        info_chunk.columns = ['#CHROM','POS','REF','ALT']
        vcf_chunk = pd.DataFrame([['.','.','.','PR','GT'] for i in range(i_end-i)],columns=vcf_head)
        vcf_chunk = pd.concat([info_chunk[['#CHROM','POS']],vcf_chunk['ID'],info_chunk[['REF','ALT']],vcf_chunk[['QUAL','FILTER','INFO','FORMAT']],pd.DataFrame(g_chunk,columns=samples)],axis=1)
        pbar.update(i_end-i)
        if i % 10 == 0:
            memory_usage = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f'{memory_usage:.2f} GB')
        if i == 0:
            vcf_chunk.to_csv(f'{outPrefix}.vcf',sep='\t',index=None,mode='a') # keep header
        else:
            vcf_chunk.to_csv(f'{outPrefix}.vcf',sep='\t',index=None,header=False,mode='a') # ignore header

if __name__ == "__main__":
    pass
