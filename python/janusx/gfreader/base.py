from importlib.metadata import version, PackageNotFoundError
from typing import Literal, Union
try:
    v = version("janusx")
except PackageNotFoundError:
    v = "0.0.0"
from .gfreader import load_genotype_chunks, inspect_genotype_file
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
import os
import sys
from time import monotonic
from janusx.script._common.status import (
    get_rich_spinner_name,
    print_success,
    print_failure,
    format_elapsed,
)

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

process = psutil.Process()
def get_process_info():
    """Return current CPU utilization and resident memory usage."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024**3  # GB
    return cpu_percent, memory_mb


class _LoadProgressAdapter:
    """
    Genotype loading progress: rich-first, tqdm fallback.
    """
    def __init__(self, total: int, desc: str, *, enabled: bool = True) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self.enabled = bool(enabled)
        self._backend = "none"
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._start_ts = monotonic()

        if (not self.enabled):
            return

        if _HAS_RICH_PROGRESS and getattr(sys.stdout, "isatty", lambda: False)():
            try:
                self._progress = Progress(
                    SpinnerColumn(
                        spinner_name=get_rich_spinner_name(),
                        style="cyan",
                        finished_text="[green]✔︎[/green]",
                    ),
                    TextColumn("[green]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                )
                self._progress.start()
                self._task_id = self._progress.add_task(self.desc, total=self.total)
                self._backend = "rich"
            except Exception:
                self._progress = None
                self._task_id = None

        if self._backend == "none":
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                ascii=True,
                leave=False,
                dynamic_ncols=True,
            )
            self._backend = "tqdm"

    def update(self, n: int) -> None:
        step = int(max(0, n))
        if step == 0:
            return
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=step)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)

    def close(self, *, success: bool = True) -> None:
        elapsed_text = format_elapsed(max(0.0, float(monotonic() - self._start_ts)))

        if self._backend == "rich" and self._progress is not None:
            if success and self._task_id is not None:
                try:
                    task = self._progress.tasks[self._task_id]
                    if not task.finished:
                        self._progress.update(self._task_id, completed=task.total)
                except Exception:
                    pass
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        if not self.enabled:
            return
        if success:
            print_success(f"{self.desc} ...Finished [{elapsed_text}]")
        else:
            print_failure(f"{self.desc} ...Failed [{elapsed_text}]")

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

def breader(
    prefix: str,
    chunk_size=10_000,
    maf: float = 0,
    miss: float = 1,
    impute: bool = False,
    dtype: Literal['int8', 'float32'] = 'int8',
    *,
    show_progress: bool = True,
    progress_desc: str = "Loading genotype",
) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    idv,m = inspect_genotype_file(prefix)
    chunks = load_genotype_chunks(prefix,chunk_size,maf,miss,impute)
    genotype = np.zeros(shape=(len(idv),m),dtype=dtype)
    pbar = _LoadProgressAdapter(m, str(progress_desc), enabled=bool(show_progress))
    num = 0
    try:
        for chunk,_ in chunks:
            cksize = chunk.shape[0]
            genotype[:,num:num+cksize] = chunk.T
            num += cksize
            pbar.update(cksize)
    except Exception:
        pbar.close(success=False)
        raise
    pbar.close(success=True)
    bim = pd.read_csv(f'{prefix}.bim',sep=r'\s+',header=None)
    genotype = pd.DataFrame(genotype,index=idv,).T
    genotype = pd.concat([bim[[0,3,4,5]],genotype],axis=1)
    genotype.columns = ['#CHROM','POS','A0','A1']+idv
    genotype = genotype.set_index(['#CHROM','POS'])
    return genotype.dropna()

def vcfreader(
    vcfPath: str,
    chunk_size=50_000,
    maf: float = 0,
    miss: float = 1,
    impute: bool = False,
    dtype: Literal['int8', 'float32'] = 'int8',
    *,
    show_progress: bool = True,
    progress_desc: str = "Loading genotype",
) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    idv,m = inspect_genotype_file(vcfPath)
    chunks = load_genotype_chunks(vcfPath,chunk_size,maf,miss,impute)
    genotype = np.zeros(shape=(len(idv),m),dtype=dtype)
    pbar = _LoadProgressAdapter(m, str(progress_desc), enabled=bool(show_progress))
    num = 0
    bim = []
    try:
        for chunk,site in chunks:
            cksize = chunk.shape[0]
            genotype[:,num:num+cksize] = chunk.T
            num += cksize
            pbar.update(cksize)
            bim.extend([[i.chrom,i.pos,i.ref_allele,i.alt_allele] for i in site])
    except Exception:
        pbar.close(success=False)
        raise
    pbar.close(success=True)
    bim = pd.DataFrame(bim)
    genotype = pd.DataFrame(genotype,index=idv,).T
    genotype = pd.concat([bim,genotype],axis=1)
    genotype.columns = ['#CHROM','POS','A0','A1']+idv
    genotype = genotype.set_index(['#CHROM','POS'])
    return genotype.dropna()


def greader(
    txtPath: str,
    delimiter: Union[str, None] = None,
    dtype: Literal['int8', 'float32'] = 'int8',
) -> pd.DataFrame:
    """
    Load genotype TXT matrix by NumPy.

    Assumption:
      - first line is sample IDs
      - remaining lines are SNP-major numeric genotypes
    """
    with open(txtPath, "r", encoding="utf-8") as f:
        header = ""
        for line in f:
            line = line.strip()
            if line:
                header = line
                break
    if header == "":
        raise ValueError(f"Empty genotype TXT file: {txtPath}")

    if delimiter is None:
        if "," in header and ("\t" not in header):
            delimiter = ","
        elif "\t" in header:
            delimiter = "\t"
        else:
            delimiter = None

    tokens = header.split(delimiter) if delimiter is not None else header.split()
    sample_ids = [str(x).strip() for x in tokens if str(x).strip() != ""]

    genotype = np.loadtxt(
        txtPath,
        dtype=dtype,
        delimiter=delimiter,
        skiprows=1,
    )
    genotype = np.atleast_2d(genotype)

    if genotype.shape[1] != len(sample_ids):
        if len(sample_ids) == genotype.shape[1] + 1:
            sample_ids = sample_ids[1:]
        else:
            sample_ids = [f"S{i+1}" for i in range(genotype.shape[1])]

    genotype = np.asarray(genotype, dtype=dtype)
    nsnp = genotype.shape[0]
    bim = pd.DataFrame(
        {
            '#CHROM': np.repeat("N", nsnp),
            'POS': np.arange(1, nsnp + 1, dtype=np.int64),
            'A0': np.repeat("N", nsnp),
            'A1': np.repeat("N", nsnp),
        }
    )
    genotype = pd.DataFrame(genotype, columns=sample_ids)
    genotype = pd.concat([bim[['#CHROM', 'POS', 'A0', 'A1']], genotype], axis=1)
    genotype = genotype.set_index(['#CHROM', 'POS'])
    return genotype.dropna()


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
