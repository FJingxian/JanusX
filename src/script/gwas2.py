# -*- coding: utf-8 -*-
"""
JanusX – High Performance GWAS Command-Line Interface

功能概览
--------
支持三种模型：
  - LMM  (pyBLUP.GWAS / slim.LMM)
  - LM   (pyBLUP.LM / slim.LM)
  - FarmCPU (pyBLUP.farmcpu)

支持两种总体运行模式：
  1. 常规模式（默认）：一次性加载所有基因型矩阵到内存，使用 QK 做过滤、GRM、PCA。
  2. 低内存模式（-lmem）：不把全矩阵加载到内存，按 chunk 从 rust2py.gfreader 流式读取。

主要步骤
--------
  1. 解析参数 & 设置日志
  2. 确定基因型输入来源（VCF / BFILE / NPY）
  3. 加载表型 & 可选列
  4. 根据模式分流：
      - 低内存模式：inspect_genotype_file + load_genotype_chunks + slim.LMM/LM
      - 常规模式：vcfreader/breader/npyreader + QK + GRM + PCA + GWAS
  5. 对每个性状依次运行 GWAS（LMM / LM / FarmCPU）
  6. 按需绘图 & 输出结果

Citation
--------
  https://github.com/MaizeMan-JxFU/JanusX/
"""

import os
import time
import socket
import argparse

# ---- matplotlib 环境设置（非交互，适合服务器） ----
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm import tqdm
import psutil

from bioplotkit import GWASPLOT
from pyBLUP import GWAS, LM, farmcpu
from pyBLUP import QK
from pyBLUP import slim
from gfreader import breader, vcfreader, npyreader
from rust2py.gfreader import load_genotype_chunks, inspect_genotype_file
from ._common.log import setup_logging


# ======================================================================
# 通用工具函数
# ======================================================================

def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """
    快速绘制 GWAS 结果（直方图 + Manhattan + QQ）。

    Parameters
    ----------
    gwasresult : pd.DataFrame
        必须包含至少 ['POS', 'p'] 等列。
    phenosub : np.ndarray
        GWAS 使用的表型值（一维）。
    xlabel : str
        表型名称，用于直方图 x 轴标题。
    outpdf : str
        输出 PDF 文件名。
    """
    results = gwasresult.astype({"POS": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    layout = [["A", "B", "B", "C"]]
    axes = fig.subplot_mosaic(mosaic=layout)

    gwasplot = GWASPLOT(results)

    # A: phenotype 分布
    axes["A"].hist(phenosub, bins=15)
    axes["A"].set_xlabel(xlabel)
    axes["A"].set_ylabel("count")

    # B: Manhattan
    gwasplot.manhattan(-np.log10(1 / results.shape[0]), ax=axes["B"])

    # C: QQ
    gwasplot.qq(ax=axes["C"])

    plt.tight_layout()
    plt.savefig(outpdf, transparent=True)


def determine_genotype_source(args) -> tuple[str, str]:
    """
    根据命令行参数确定基因型文件路径和输出前缀。

    Returns
    -------
    gfile : str
        基因型文件前缀 / 路径。
    prefix : str
        输出文件前缀。
    """
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    elif args.npy:
        gfile = args.npy
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input specified. Use -vcf / -bfile / -npy.")

    if args.prefix is not None:
        prefix = args.prefix

    # 统一路径分隔符
    gfile = gfile.replace("\\", "/")
    return gfile, prefix


def load_phenotype(phenofile: str, ncol: list[int] | None, logger) -> pd.DataFrame:
    """
    加载并整理表型，按需子集列。

    - 第一列为样本 ID
    - 对重复 ID 取平均

    Parameters
    ----------
    phenofile : str
        表型文件路径（tab 分隔）。
    ncol : list[int] or None
        欲分析的表型列索引（0-based）。None 则使用全部列。
    logger : logging.Logger
        日志记录器。

    Returns
    -------
    pheno : pd.DataFrame
        index 为样本 ID，列为表型。
    """
    logger.info(f"Loading phenotype from {phenofile}...")
    pheno = pd.read_csv(phenofile, sep="\t")
    pheno = pheno.groupby(pheno.columns[0]).mean()
    pheno.index = pheno.index.astype(str)

    assert pheno.shape[1] > 0, (
        "No phenotype data found, please check the phenotype file format!\n"
        f"{pheno.head()}"
    )

    if ncol is not None:
        assert np.min(ncol) < pheno.shape[1], "Phenotype column index out of range."
        ncol = [i for i in ncol if i in range(pheno.shape[1])]
        logger.info("These phenotypes will be analyzed: " +
                    "\t".join(pheno.columns[ncol]))
        pheno = pheno.iloc[:, ncol]

    return pheno


# ======================================================================
# 低内存模式：GRM / PCA / GWAS（slim.LMM / slim.LM）
# ======================================================================

def build_grm_lowmem(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mgrm: str,
) -> tuple[np.ndarray, int]:
    """
    在低内存模式下，基于流式分块构建 GRM。

    Parameters
    ----------
    genofile : str
        基因型文件路径。
    n_samples : int
        样本数。
    n_snps : int
        SNP 总数（inspect_genotype_file 返回）。
    maf_threshold : float
        MAF 阈值。
    max_missing_rate : float
        缺失率阈值。
    chunk_size : int
        分块大小。
    mgrm : str
        '1' => VanRaden, '2' => Yang 等。

    Returns
    -------
    grm : np.ndarray
        n x n 的 GRM。
    eff_m : int
        实际参与计算的 SNP 数。
    """
    method_int = int(mgrm)
    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = tqdm(total=n_snps, desc="Build GRM (lowmem)", ascii=True)
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0

    for genosub, _sites in load_genotype_chunks(
        genofile, chunk_size, maf_threshold, max_missing_rate
    ):
        genosub: np.ndarray = genosub  # (m_chunk, n_samples)
        maf = genosub.mean(axis=1, dtype="float32", keepdims=True) / 2
        genosub = genosub - 2 * maf

        if method_int == 1:
            grm += genosub.T @ genosub
            varsum += np.sum(2 * maf * (1 - maf))
        elif method_int == 2:
            grm += 1 / (2 * maf * (1 - maf)) * genosub.T @ genosub

        eff_m += genosub.shape[0]
        pbar.update(genosub.shape[0])

        if eff_m % (10 * chunk_size) == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    pbar.close()

    if method_int == 1:
        grm = (grm + grm.T) / varsum / 2
    elif method_int == 2:
        grm = (grm + grm.T) / eff_m / 2

    return grm, eff_m


def build_pcs_from_grm(grm: np.ndarray, dim: int) -> np.ndarray:
    """
    从 GRM 计算前 dim 个主成分（eigendecomposition）。

    Parameters
    ----------
    grm : np.ndarray
        n x n GRM。
    dim : int
        主成分数。

    Returns
    -------
    pcs : np.ndarray
        n x dim PC 矩阵。
    """
    eigval, eigvec = np.linalg.eigh(grm)
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, idx]
    return eigvec[:, :dim]


def load_or_build_pcs_lowmem(
    grm: np.ndarray,
    pcdim: str,
) -> np.ndarray:
    """
    在低内存模式下，根据 pcdim 构建或加载 Q 矩阵（PC）。

    pcdim 情况：
      - '0'               => 不使用 PC（返回 n x 0 空矩阵）
      - 'k' (1..n-1)      => 通过 GRM 计算前 k 个 PC
      - 其他存在的文件路径 => 外部 Q 矩阵文件

    Returns
    -------
    pcs : np.ndarray
        n x q 矩阵。
    """
    n = grm.shape[0]

    if pcdim in np.arange(1, n).astype(str):
        dim = int(pcdim)
        return build_pcs_from_grm(grm, dim)
    elif pcdim == "0":
        return np.zeros((n, 0), dtype="float32")
    elif os.path.isfile(pcdim):
        pcs = np.genfromtxt(pcdim, dtype="float32")
        assert pcs.shape[0] == n, (
            f"PCA file size not match: expected {n}, got {pcs.shape[0]}"
        )
        return pcs
    else:
        raise ValueError(f"Unknown PCA option: {pcdim}")


def run_lowmem_gwas(
    genofile: str,
    pheno_file: str,
    pheno_col: list[int] | None,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    plot: bool,
    mgrm: str,
    pcdim: str,
    model: str,
    threads: int,
) -> None:
    """
    低内存模式下的 GWAS 主流程，只支持 LMM/LM（slim.LMM / slim.LM）。

    流程：
      1. inspect_genotype_file 得到 ids, m
      2. build_grm_lowmem + load_or_build_pcs_lowmem
      3. 对每个表型：
         - 对齐样本
         - 用 slim.LMM/LM 初始化模型
         - 分块 load_genotype_chunks 计算 GWAS
    """
    # 1) phenotype
    pheno = pd.read_csv(pheno_file, sep="\t")
    pheno = pheno.groupby(pheno.columns[0]).mean()
    pheno.index = pheno.index.astype(str)

    if pheno_col is not None:
        assert np.min(pheno_col) < pheno.shape[1], "Phenotype column index out of range."
        pheno_col = [i for i in pheno_col if i in range(pheno.shape[1])]
        pheno = pheno.iloc[:, pheno_col]

    # 2) genotype meta (ids, total snps)
    ids, m_total = inspect_genotype_file(genofile)
    ids = np.array(ids).astype(str)
    n_samples = len(ids)

    # 3) GRM
    if mgrm in ["1", "2"]:
        grm, m_eff = build_grm_lowmem(
            genofile,
            n_samples,
            m_total,
            maf_threshold,
            max_missing_rate,
            chunk_size,
            mgrm,
        )
    elif os.path.isfile(mgrm):
        grm = np.genfromtxt(mgrm, dtype="float32")
        assert grm.size == n_samples * n_samples, (
            f"GRM file size not match: expected {n_samples*n_samples}, got {grm.size}"
        )
        grm = grm.reshape(n_samples, n_samples)
        m_eff = m_total
    else:
        raise ValueError(f"Unknown GRM option: {mgrm}")

    # 4) Q / PC matrix
    qmatrix = load_or_build_pcs_lowmem(grm, pcdim)

    # 5) model class
    modelmap = {"lmm": slim.LMM, "lm": slim.LM}
    ModelCls = modelmap[model]

    process = psutil.Process()

    for pname in pheno.columns:
        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        pheno_vec = pheno_sub.loc[ids[sameidx]].values

        if model == "lmm":
            mod = ModelCls(
                y=pheno_vec,
                X=qmatrix[sameidx],
                kinship=grm[sameidx][:, sameidx],
            )
        else:
            mod = ModelCls(y=pheno_vec, X=qmatrix[sameidx])

        print("*" * 60)
        if model == "lmm":
            print(
                f"Number of samples: {np.sum(sameidx)}, "
                f"Number of SNP: {m_eff}, pve of null: {round(mod.pve, 3)}"
            )

        results_chunks = []
        maf_list = []
        info_list = []
        num_done = 0

        pbar = tqdm(total=m_eff, desc=f"{model}-{pname}", ascii=True)

        for genosub, sites in load_genotype_chunks(
            genofile, chunk_size, maf_threshold, max_missing_rate
        ):
            genosub = genosub[:, sameidx]  # (m_chunk, n_use)
            maf_list.extend(np.mean(genosub, axis=1) / 2)

            # mod.gwas 返回 [beta, se, p]
            results_chunks.append(mod.gwas(genosub, threads=threads))
            info_list.extend(
                [[s.chrom, s.pos, s.ref_allele, s.alt_allele] for s in sites]
            )

            m_chunk = genosub.shape[0]
            pbar.update(m_chunk)
            num_done += m_chunk

            if num_done % (10 * chunk_size) == 0:
                mem = process.memory_info().rss / 1024**3
                pbar.set_postfix(memory=f"{mem:.2f} GB")

        pbar.close()

        # 整合结果
        results = np.concatenate(results_chunks, axis=0)
        info_arr = np.array(info_list)

        df = pd.DataFrame(
            np.concatenate(
                [info_arr, results, np.array(maf_list).reshape(-1, 1)], axis=1
            ),
            columns=["#CHROM", "POS", "REF", "ALT", "beta", "se", "p", "maf"],
        )
        df = df[["#CHROM", "POS", "REF", "ALT", "maf", "beta", "se", "p"]]
        df = df.astype(
            {"POS": int, "maf": float, "beta": float, "se": float, "p": float}
        )

        if plot:
            fastplot(
                df,
                pheno_vec,
                xlabel=pname,
                outpdf=f"{outprefix}.{pname}.{model}.pdf",
            )

        df = df.astype({"p": "object"})
        df.loc[:, "p"] = df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outprefix}.{pname}.{model}.tsv"
        df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        print(f"Saved in {out_tsv}".replace("//", "/"))


# ======================================================================
# 常规模式：一次性加载基因型 + QK + GWAS (GWAS/LM/farmcpu)
# ======================================================================

def load_genotype_full(args, gfile: str, logger):
    """
    一次性加载基因型数据，返回:
      - ref_alt : DataFrame 前两列 (chr,pos)
      - famid   : 样本 ID 数组
      - geno    : SNP x N 矩阵（numpy）
    """
    if args.vcf:
        logger.info(f"** Loading genotype from {gfile}...")
        geno_df = vcfreader(gfile)
    elif args.bfile:
        logger.info(f"Loading genotype from {gfile}.bed...")
        geno_df = breader(gfile)
    elif args.npy:
        logger.info(f"Loading genotype from {gfile}.npz...")
        geno_df = npyreader(gfile)
    else:
        raise ValueError("No genotype input specified.")

    ref_alt = geno_df.iloc[:, :2]
    famid = geno_df.columns[2:].values.astype(str)
    geno = geno_df.iloc[:, 2:].to_numpy(copy=False)

    return ref_alt, famid, geno


def prepare_qk_and_filter(geno: np.ndarray, ref_alt: pd.DataFrame, logger):
    """
    使用 QK 对基因型做过滤和 impute，并同步更新 ref_alt（chr, pos, maf）。

    返回：
      geno_filtered, ref_alt_filtered, qkmodel
    """
    logger.info(
        "* Filter SNPs with MAF < 0.01 or missing rate > 0.05; impute with mode..."
    )
    logger.info("Recommended: use genotype matrix imputed by beagle or impute2 as input")

    qkmodel = QK(geno, maff=0.01)
    logger.info("Filter finished")
    geno_filtered = qkmodel.M

    ref_alt_filt = ref_alt.loc[qkmodel.SNPretain].copy()
    # 对于 MAF 非常小的标记，ref/alt 翻转
    ref_alt_filt.iloc[qkmodel.maftmark, [0, 1]] = ref_alt_filt.iloc[
        qkmodel.maftmark, [1, 0]
    ]
    ref_alt_filt["maf"] = qkmodel.maf

    return geno_filtered, ref_alt_filt, qkmodel


def build_grm_full(
    args,
    gfile_prefix: str,
    qkmodel: QK,
    kinship_method: str,
    logger,
) -> np.ndarray:
    """
    在常规模式下构建 / 加载 GRM。

    - 若 kinship_method in ['1','2']，则调用 QK.GRM 并缓存为 {prefix}.k.{method}.txt
    - 否则从外部文件加载（支持 .npz）
    """
    if args.lmm:
        logger.info("* Preparing GRM for LMM...")
        if kinship_method in ["1", "2"]:
            km_path = f"{gfile_prefix}.k.{kinship_method}.txt"
            if os.path.exists(km_path):
                logger.info(f"* Loading GRM from {km_path}...")
                kmatrix = np.genfromtxt(km_path)
            else:
                logger.info(
                    f"* Calculation method of kinship matrix is {kinship_method}"
                )
                kmatrix = qkmodel.GRM(method=int(kinship_method))
                np.savetxt(km_path, kmatrix, fmt="%.6f")
        else:
            logger.info(f"* Loading GRM from {kinship_method}...")
            if kinship_method.endswith(".npz"):
                kmatrix = np.load(kinship_method)["arr_0"]
            else:
                kmatrix = np.genfromtxt(kinship_method)
        logger.info(f"GRM {str(kmatrix.shape)}:")
        logger.info(kmatrix[:5, :5])
        return kmatrix
    else:
        return None


def build_qmatrix_full(
    args,
    gfile_prefix: str,
    qkmodel: QK,
    geno: np.ndarray,
    qdim: str,
    cov_path: str | None,
    csnp: str | None,
    ref_alt: pd.DataFrame,
    logger,
) -> np.ndarray:
    """
    构建或加载 Q 矩阵（PC + 可选协变量 + 条件 SNP）。

    返回：
      qmatrix : N x q
    """
    # Q matrix: PCA 部分
    if qdim in np.arange(0, 30).astype(str):
        q_path = f"{gfile_prefix}.q.{qdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"* Loading Q matrix from {q_path}...")
            qmatrix = np.genfromtxt(q_path)
        elif qdim == "0":
            qmatrix = np.array([]).reshape(geno.shape[1], 0)
        else:
            logger.info(f"* Dimension of PC for Q matrix is {qdim}")
            qmatrix, _eigval = qkmodel.PCA()
            qmatrix = qmatrix[:, : int(qdim)]
            np.savetxt(q_path, qmatrix, fmt="%.6f")
    else:
        logger.info(f"* Loading Q matrix from {qdim}...")
        qmatrix = np.genfromtxt(qdim)

    # 额外 covariate 文件
    if cov_path:
        cov_arr = np.genfromtxt(cov_path).reshape(-1, 1)
        logger.info(f"Covmatrix {cov_arr.shape}:")
        qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)

    # 条件 SNP（conditional GWAS）
    if csnp:
        logger.info(f"* Use SNP in {csnp} as control for conditional GWAS")
        chr_loc_index = (
            ref_alt.reset_index()
            .iloc[:, :2]
            .astype(str)
        )
        chr_loc_index = pd.Index(
            chr_loc_index.iloc[:, 0] + ":" + chr_loc_index.iloc[:, 1]
        )
        cov_vec = geno[chr_loc_index.get_loc(csnp)].reshape(-1, 1)
        logger.info(f"Covmatrix {cov_vec.shape}:")
        qmatrix = np.concatenate([qmatrix, cov_vec], axis=1)

    logger.info(f"Qmatrix {str(qmatrix.shape)}:")
    logger.info(qmatrix[:5, :5])

    return qmatrix


def run_full_gwas_for_trait(
    phename: str,
    p_series: pd.Series,
    famid: np.ndarray,
    geno: np.ndarray,
    ref_alt: pd.DataFrame,
    qmatrix: np.ndarray,
    kmatrix: np.ndarray | None,
    outfolder: str,
    prefix: str,
    args,
    logger,
):
    """
    对单个表型运行 GWAS（三种模型之一）。

    - 根据 famid 与 phenotype 对齐样本
    - 视 model 选择 GWAS / LM / FarmCPU
    - 输出 TSV & 可选 PDF
    """
    t = time.time()
    logger.info("*" * 60)
    logger.info(f"* GWAS process for {phename}")

    p = p_series.dropna()
    famidretain = np.isin(famid, p.index)

    if np.sum(famidretain) == 0:
        logger.info(
            f"Phenotype {phename} has no overlapping samples with genotype, "
            "please check sample id. skipped.\n"
        )
        return

    snp_sub = geno[:, famidretain]
    p_sub = p.loc[famid[famidretain]].values.reshape(-1, 1)
    q_sub = qmatrix[famidretain]

    # ---- 1) LMM ----
    if args.lmm:
        logger.info("** Mixed Linear Model:")
        k_sub = kmatrix[famidretain][:, famidretain]
        gwasmodel = GWAS(y=p_sub, X=q_sub, kinship=k_sub)
        logger.info(
            f"Number of samples: {np.sum(famidretain)}, "
            f"Number of SNP: {geno.shape[0]}, "
            f"pve of null: {round(gwasmodel.pve, 3)}"
        )

        res = gwasmodel.gwas(snp=snp_sub, chunksize=100_000, threads=args.thread)
        res_df = pd.DataFrame(
            res, columns=["beta", "se", "p"], index=ref_alt.index
        )
        res_df = pd.concat([ref_alt, res_df], axis=1)
        res_df = res_df.reset_index().dropna()

        logger.info(f"Effective number of SNP: {res_df.shape[0]}")

        if args.plot:
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.lmm.pdf",
            )

        res_df = res_df.astype({"p": "object"})
        res_df.loc[:, "p"] = res_df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.lmm.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=False)
        logger.info(f"Saved in {out_tsv}".replace("//", "/"))

    # ---- 2) LM ----
    if args.lm:
        logger.info("** General Linear Model:")
        gwasmodel = LM(y=p_sub, X=q_sub)

        res = gwasmodel.gwas(snp=snp_sub, chunksize=100_000, threads=args.thread)
        res_df = pd.DataFrame(
            res, columns=["beta", "se", "p"], index=ref_alt.index
        )
        res_df = pd.concat([ref_alt, res_df], axis=1)
        res_df = res_df.reset_index().dropna()

        if args.plot:
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.lm.pdf",
            )

        res_df = res_df.astype({"p": "object"})
        res_df.loc[:, "p"] = res_df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.lm.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        logger.info(f"Saved in {out_tsv}".replace("//", "/"))

    # ---- 3) FarmCPU ----
    if args.farmcpu:
        logger.info("** FarmCPU Model:")
        res = farmcpu(
            y=p_sub,
            M=snp_sub,
            X=q_sub,
            chrlist=ref_alt.reset_index().iloc[:, 0].values,
            poslist=ref_alt.reset_index().iloc[:, 1].values,
            iter=20,
            threads=args.thread,
        )
        res_df = pd.DataFrame(
            res, columns=["beta", "se", "p"], index=ref_alt.index
        )
        res_df = pd.concat([ref_alt, res_df], axis=1)
        res_df = res_df.reset_index()

        if args.plot:
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.farmcpu.pdf",
            )

        res_df = res_df.astype({"p": "object"})
        res_df.loc[:, "p"] = res_df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.farmcpu.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        logger.info(f"Saved in {out_tsv}".replace("//", "/"))

    logger.info(f"Time costed: {round(time.time() - t, 2)} secs\n")


def run_fullmem_gwas(args, gfile: str, prefix: str, logger):
    """
    常规内存模式主流程：
      1. 加载 phenotype
      2. 加载 genotype 全矩阵
      3. QK 过滤 / impute
      4. 构建 / 加载 GRM, Q 矩阵
      5. 遍历表型，分别跑 LMM/LM/FarmCPU
    """
    t_loading = time.time()
    phenofile = args.pheno
    outfolder = args.out
    kinship_method = args.grm
    qdim = args.qcov
    cov = args.cov

    logger.info("* Loading genotype and phenotype")
    if not args.npy:
        logger.info(
            "Recommended: use NumPy format of genotype matrix (use gformat module to convert)"
        )

    # 1) phenotype
    pheno = load_phenotype(phenofile, args.ncol, logger)

    # 2) genotype
    ref_alt, famid, geno = load_genotype_full(args, gfile, logger)
    logger.info(
        f"Geno and Pheno are ready, costed {(time.time() - t_loading):.2f} secs"
    )

    # 3) QK filter / impute
    geno, ref_alt, qkmodel = prepare_qk_and_filter(geno, ref_alt, logger)

    # dominance 模式（添加型 GRM + 显性 SNP 效应）
    if args.dom:
        logger.info("* Transfer additive gmatrix to dominance gmatrix")
        np.subtract(geno, 1, out=geno)
        np.absolute(geno, out=geno)

    # 4) 若限制染色体区间
    if args.chrloc:
        chr_loc = np.array(args.chrloc.split(":"), dtype=np.int32)
        chr_id, start, end = chr_loc[0], np.min(chr_loc[1:]), np.max(chr_loc[1:])
        onlySNP = ref_alt.index.to_frame().values
        filt_chr = onlySNP[:, 0].astype(str) == str(chr_id)

        if start == 0 and end == 0:
            geno = geno[filt_chr]
            ref_alt = ref_alt.loc[filt_chr]
        else:
            filt_pos = (onlySNP[filt_chr, 1] <= end) & (
                onlySNP[filt_chr, 1] >= start
            )
            geno = geno[filt_chr][filt_pos]
            ref_alt = ref_alt.loc[filt_chr].loc[filt_pos]

    assert geno.size > 0, "After filtering, number of SNP is 0"

    gfile_prefix = gfile.replace(".vcf", "").replace(".gz", "")

    # 5) LMM GRM
    kmatrix = build_grm_full(args, gfile_prefix, qkmodel, kinship_method, logger)

    # 6) Q matrix (PC + cov + csnp)
    qmatrix = build_qmatrix_full(
        args,
        gfile_prefix,
        qkmodel,
        geno,
        qdim,
        cov,
        args.csnp,
        ref_alt,
        logger,
    )

    del qkmodel

    # 7) 对每个 phenotype 跑 GWAS
    for phename in pheno.columns:
        run_full_gwas_for_trait(
            phename=phename,
            p_series=pheno[phename],
            famid=famid,
            geno=geno,
            ref_alt=ref_alt,
            qmatrix=qmatrix,
            kmatrix=kmatrix,
            outfolder=outfolder,
            prefix=prefix,
            args=args,
            logger=logger,
        )


# ======================================================================
# CLI：参数解析 & 主入口
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 必需参数
    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf",
        "--vcf",
        type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz)",
    )
    geno_group.add_argument(
        "-bfile",
        "--bfile",
        type=str,
        help="Input genotype files in PLINK binary format "
        "(prefix for .bed, .bim, .fam)",
    )
    geno_group.add_argument(
        "-npy",
        "--npy",
        type=str,
        help="Input genotype in NumPy format "
        "(prefix for .npz/.snp/.idv from gformat)",
    )

    required_group.add_argument(
        "-p",
        "--pheno",
        type=str,
        required=True,
        help="Phenotype file (tab-delimited, sample IDs in first column)",
    )

    # 模型参数
    models_group = parser.add_argument_group("Model Arguments")
    models_group.add_argument(
        "-lmm",
        "--lmm",
        action="store_true",
        default=False,
        help="Linear mixed model (default: %(default)s)",
    )
    models_group.add_argument(
        "-farmcpu",
        "--farmcpu",
        action="store_true",
        default=False,
        help="FarmCPU model (default: %(default)s)",
    )
    models_group.add_argument(
        "-lm",
        "--lm",
        action="store_true",
        default=False,
        help="General linear model (default: %(default)s)",
    )

    # 可选参数
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n",
        "--ncol",
        action="extend",
        nargs="*",
        default=None,
        type=int,
        help='Analyzed phenotype column indices (0-based). '
             'E.g., "-n 0 -n 3" to analyze phenotype 1 and 4 '
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-cl",
        "--chrloc",
        type=str,
        default=None,
        help='Only analyze SNPs in a specific region, e.g., "1:1000000:3000000" '
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-lmem",
        "--lmem",
        action="store_true",
        default=False,
        help="Low memory mode (stream genotypes from disk; FarmCPU not supported) "
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-k",
        "--grm",
        type=str,
        default="1",
        help="Kinship matrix calculation method [1-centralization or 2-standardization] "
             "or path to pre-calculated GRM file "
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-q",
        "--qcov",
        type=str,
        default="0",
        help="Number of principal components for Q matrix or path to Q matrix file "
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-c",
        "--cov",
        type=str,
        default=None,
        help="Path to additional covariate file (default: %(default)s)",
    )
    optional_group.add_argument(
        "-d",
        "--dom",
        action="store_true",
        default=False,
        help="Estimate dominance effects (default: %(default)s)",
    )
    optional_group.add_argument(
        "-csnp",
        "--csnp",
        type=str,
        default=None,
        help='Control SNP for conditional GWAS, e.g., "1:1200000" '
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-plot",
        "--plot",
        action="store_true",
        default=False,
        help="Visualization of GWAS results (default: %(default)s)",
    )
    optional_group.add_argument(
        "-t",
        "--thread",
        type=int,
        default=-1,
        help="Number of CPU threads to use (-1 for all available cores, "
             "default: %(default)s)",
    )
    optional_group.add_argument(
        "-o",
        "--out",
        type=str,
        default=".",
        help="Output directory for results (default: %(default)s)",
    )
    optional_group.add_argument(
        "-prefix",
        "--prefix",
        type=str,
        default=None,
        help="Prefix of output files (default: %(default)s)",
    )

    return parser.parse_args()


def main(log: bool = True):
    t_start = time.time()
    args = parse_args()

    # 线程数
    if args.thread <= 0:
        args.thread = cpu_count()

    # 基因型输入和前缀
    gfile, prefix = determine_genotype_source(args)

    # 输出目录 & 日志
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{args.out}/{prefix}.gwas.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

    logger.info(
        "High Performance Linear Mixed Model Solver for Genome-Wide Association Studies"
    )
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("GWAS LMM SOLVER CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        if args.chrloc:
            logger.info(f"Analysis nSNP:    {args.chrloc}")
        else:
            logger.info("Analysis nSNP:    All")
        if args.ncol is not None:
            logger.info(f"Analysis Pcol:    {args.ncol}")
        else:
            logger.info("Analysis Pcol:    All")
        if args.lm:
            logger.info("Estimate Model:   General Linear model")
        if args.lmm:
            logger.info("Estimate Model:   Mixed Linear model")
        if args.farmcpu:
            logger.info("Estimate Model:   FarmCPU")
        logger.info(f"Low memory mode:  {args.lmem}")
        if args.dom:
            logger.info(f"Dominance model:  {args.dom}")
        if args.csnp:
            logger.info(f"Conditional SNP:  {args.csnp}")
        logger.info(f"Estimate of GRM:  {args.grm}")
        if args.qcov != "0":
            logger.info(f"Q matrix:         {args.qcov}")
        if args.cov:
            logger.info(f"Covariant matrix: {args.cov}")
        logger.info(f"Threads:          {args.thread} ({cpu_count()} available)")
        logger.info(f"Output prefix:    {args.out}/{prefix}")
        logger.info("*" * 60 + "\n")

    # 一些基础检查
    try:
        assert os.path.isfile(args.pheno), f"cannot find phenotype file {args.pheno}"
        kcal = args.grm in ["1", "2"] or os.path.isfile(args.grm)
        qcal = args.qcov in np.arange(0, 30).astype(str) or os.path.isfile(args.qcov)
        assert kcal, f"{args.grm} is not a calculation method or GRM file"
        assert qcal, f"{args.qcov} is not a dimension of PC or PC file"
        assert args.cov is None or os.path.isfile(args.cov), (
            f"{args.cov} is applied, but it is not a file"
        )
        assert (
            args.lm or args.lmm or args.farmcpu
        ), "No model to estimate, try -lm, -farmcpu or -lmm"

        # 低内存模式
        if args.lmem:
            if args.farmcpu:
                logger.info(
                    "Low memory mode does not support FarmCPU, FarmCPU will be ignored."
                )

            # 只支持 lmm / lm
            if args.lmm:
                run_lowmem_gwas(
                    genofile=gfile,
                    pheno_file=args.pheno,
                    pheno_col=args.ncol,
                    outprefix=f"{args.out}/{prefix}",
                    maf_threshold=0.01,
                    max_missing_rate=0.05,
                    chunk_size=100_000,
                    plot=args.plot,
                    mgrm=args.grm,
                    pcdim=args.qcov,
                    model="lmm",
                    threads=args.thread,
                )
            if args.lm:
                run_lowmem_gwas(
                    genofile=gfile,
                    pheno_file=args.pheno,
                    pheno_col=args.ncol,
                    outprefix=f"{args.out}/{prefix}",
                    maf_threshold=0.01,
                    max_missing_rate=0.05,
                    chunk_size=100_000,
                    plot=args.plot,
                    mgrm=args.grm,
                    pcdim=args.qcov,
                    model="lm",
                    threads=args.thread,
                )

        # 常规模式
        else:
            run_fullmem_gwas(args, gfile, prefix, logger)

    except Exception as e:
        logger.exception(f"Error of JanusX: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished, Total time: {round(time.time() - t_start, 2)} secs\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()