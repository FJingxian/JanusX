import os
import sys
import numpy as np
from gfreader_rs import save_genotype_streaming, SiteInfo
def simulate_chunks(nsnp, nidv, chunk_size=50_000, maf_low=0.02, maf_high=0.45, seed=1):
    rng = np.random.default_rng(seed)
    n_done = 0
    while n_done < nsnp:
        m = min(chunk_size, nsnp - n_done)
        mafs = rng.uniform(maf_low, maf_high, size=m).astype(np.float32)
        u = rng.random((m, nidv), dtype=np.float32)

        p = mafs
        p0 = (1 - p) ** 2
        p1 = p0 + 2 * p * (1 - p)

        g = np.empty((m, nidv), dtype=np.int8)
        g[u < p0[:, None]] = 0
        g[(u >= p0[:, None]) & (u < p1[:, None])] = 1
        g[u >= p1[:, None]] = 2
        sites = [SiteInfo('1',i,'A','T') for i in range(n_done,n_done+m)]
        yield g, sites
        n_done += m

def main():
    if len(sys.argv)>1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print('Usage: jx sim [nsnp(k)] [nidv] [outprefix]')
        elif len(sys.argv) == 4:
            nsnp, nidv = int(1e3*int(sys.argv[1])),int(sys.argv[2])
            outprefix = sys.argv[3]
            os.makedirs(os.path.dirname(outprefix),exist_ok=True,mode=0o777)
            chunks = simulate_chunks(nsnp, nidv)
            samples = np.arange(1,1+nidv).astype(str)
            y = 100+1e-3*np.zeros(shape=(nidv,1),dtype='float32')
            for g,_ in chunks:
                beta = 0.1*np.random.randn(g.shape[0],1)
                y += (beta.T@g).T
                if iter == 10:
                    y += (np.abs(np.random.randn(g.shape[0],1)).T@g).T
            chunks = simulate_chunks(nsnp, nidv)
            print(f'Generating with {nsnp:.1e} SNPs and {nidv} individuals...')
            save_genotype_streaming(outprefix,samples.tolist(),chunks,total_snps=nsnp)
            y = np.concatenate([samples.reshape(-1,1),samples.reshape(-1,1),y],axis=1,dtype=object)
            ynew = np.concatenate([[['tag','test']],y[:,[1,2]]],dtype=object)
            np.savetxt(f'{outprefix}.pheno',y,delimiter='\t',fmt=['%s', '%s', '%.3f'])
            np.savetxt(f'{outprefix}.pheno.txt',ynew,delimiter='\t',fmt=['%s', '%s'])
        else:
            print('Usage: jx sim [nsnp(k)] [nidv] [outprefix]')
    else:
        print('Usage: jx sim [nsnp(k)] [nidv] [outprefix]')