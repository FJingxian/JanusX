# JanusX

[简体中文(推荐)](./README_zh.md) | [English](../README.md) | [Zea Eureka](https://mp.weixin.qq.com/s/jl3h2DPRC21l8QJ0WxzXDA)

## Overview

JanusX 是一个高性能的一体化数量遗传分析工具，统一了全基因组关联分析（GWAS）与基因组选择（GS）。内置 GLM、LMM、fastLMM、FarmCPU 等常用 GWAS 方法，以及 GBLUP、rrBLUP、BayesA/B/Cpi 等 GS 模型，并覆盖从数据处理到可视化的完整流程。

与 [GEMMA](https://github.com/genetics-statistics/GEMMA)、[GCTA](https://yanglab.westlake.edu.cn/software/gcta)、[rMVP](https://github.com/xiaolei-lab/rMVP) 相比，JanusX 在多线程计算上有明显性能优势。

## Table of Contents

- [JanusX](#janusx)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [From Source (Latest)](#from-source-latest)
    - [PyPI (Stable)](#pypi-stable)
    - [Pre-compiled Releases (Stable)](#pre-compiled-releases-stable)
  - [Modules](#modules)
  - [Input File Formats](#input-file-formats)
    - [Phenotype File](#phenotype-file)
    - [Genotype Files](#genotype-files)
    - [GWAS Optional Matrices (GRM / Q / Covariate)](#gwas-optional-matrices-grm--q--covariate)
  - [CLI Reference](#cli-reference)
    - [GWAS (`gwas`)](#gwas-gwas)
    - [Genomic Selection (`gs`)](#genomic-selection-gs)
    - [postGWAS (`postGWAS`)](#postgwas-postgwas)
    - [Simulation (`sim`)](#simulation-sim)
  - [Citation](#citation)

## Quick Start

**GWAS 分析**:

```bash
jx gwas -bfile data -p pheno.txt -lmm
jx gwas -vcf data.vcf.gz -p pheno.txt -lmm
```

**基因组选择**:

```bash
jx gs -vcf data.vcf.gz -p pheno.txt -GBLUP
```

**postGWAS**:

```bash
# Generate Manhattan and QQ plots
jx postGWAS -f result.lmm.tsv
# With SNP annotation
jx postGWAS -f results/test.lmm.tsv -a annotation.gff -ab 30
```

![Manhattan and QQ plots](../fig/test0.png "Simple visualization")

截图使用的数据在 `example/` 目录（来源：[genetics-statistics/GEMMA](https://github.com/genetics-statistics/GEMMA)）。

**群体结构**:

```bash
# Compute GRM
jx grm -bfile data.vcf.gz
# PCA
jx pca -bfile data.vcf.gz
```

## Requirements

- Python 3.9+
- Rust toolchain（用于源码构建，maturin/PyO3）

## Installation

### From Source (Latest)

使用 Python 构建：

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
# build
pip install .
# test
jx -h
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -o test -lm -lmm -fastlmm -farmcpu -n 0 -plot
jx postGWAS -f test/mouse_hs1940.test0.lmm.tsv -threshold 1e-6 -format pdf -o test
```

使用 Docker 构建：

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
# build
docker build -t janusx:latest .
# test
docker run -v .:/mnt --rm janusx:latest -h
docker run -v .:/mnt --rm janusx:latest gwas -vcf /mnt/example/mouse_hs1940.vcf.gz -p /mnt/example/mouse_hs1940.pheno -o /mnt/test -lm -lmm -fastlmm -farmcpu -n 0 -plot
docker run -v .:/mnt --rm janusx:latest postGWAS -f /mnt/test/*.lmm.tsv -threshold 1e-6 -format pdf -o /mnt/test
```

### PyPI (Stable)

```bash
pip install janusx
jx -h
```

### Pre-compiled Releases (Stable)

我们在 [GitHub Releases](https://github.com/FJingxian/JanusX/releases) 提供 Windows（Windows11 构建）和 Linux（Ubuntu20.04 构建）的预编译版本。
下载并解压后可直接运行。

## Modules

| Module | Description |
| --- | --- |
| `gwas` | 统一的 GWAS 包装（GLM/LMM/fastLMM/FarmCPU） |
| `gs` | 基因组选择（GBLUP、rrBLUP、BayesA/B/Cpi） |
| `postGWAS` | 可视化与注释 |
| `grm` | 亲缘关系矩阵（GRM） |
| `pca` | 主成分分析 |
| `sim` | 基因型与表型模拟 |

## Input File Formats

### Phenotype File

制表符分隔。第一列为样本 ID；后续列为表型值。

```text
samples	trait1	trait2	trait3
indv1	10.5	0.85	0.05
indv2	12.3	1.00	0.08
indv3	8.6	0.92	0.04
```

### Genotype Files

- **VCF**: `.vcf` 或 `.vcf.gz`

```text
##fileformat=VCFv4.2
##fileDate=20251201
##source=JanusXv1.0.10
##contig=<ID=1,length=2>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	A	B	C	D	E	F
1	1	10000235	A	C	.	.	PR	GT	0/1	0/0	0/0	0/0	0/0	0/1
1	1	10000345	A	G	.	.	PR	GT	0/0	0/0	0/0	0/0	1/1	1/1
1	1	10004575	G	.	.	.	PR	GT	0/0	0/0	0/0	0/0	0/0	0/0
1	1	10006974	C	T	.	.	PR	GT	0/0	0/0	0/1	1/1	0/1	1/1
1	1	10006986	A	G	.	.	PR	GT	0/0	0/0	0/1	./.	1/1	1/1
```

- **PLINK**: `.bed`/`.bim`/`.fam`（使用前缀，格式说明见 http://zzz.bwh.harvard.edu/plink/data.shtml#bed）

### GWAS Optional Matrices (GRM / Q / Covariate)

- **GRM**: A 或 G 矩阵（个体间亲缘关系）。

```text
1.00	0.12	0.05
0.12	1.00	0.08
0.05	0.08	1.00
```

- **Q**: 来自 admixture 或 PCA 的 Q 矩阵。

```text
0.12	-0.03
-0.05	0.08
0.02	-0.01
```

- **Covariate**: 性别、环境或日期等协变量。

```text
0	1
1	1
0	2
```

传入 `--grm`|`-k`、`--qcov`|`-q`、`--cov`|`-c` 的文件必须为纯数值矩阵，且与基因型样本顺序一致；以空格或制表符分隔，不含行名与列名。

## CLI Reference

### GWAS (`gwas`)

统一的 GWAS 运行器，支持 GLM、LMM、fastLMM（固定 lambda）和 FarmCPU。

```bash
# LM / LMM (estimated variance parameter per snp) / fastLMM (fixed variance parameter) are streaming-friendly
jx gwas -bfile data -p pheno.txt -lm -o out
jx gwas -bfile data -p pheno.txt -lmm -o out
jx gwas -bfile data -p pheno.txt -fastlmm -o out

# FarmCPU uses full-memory to load genotype matrix
jx gwas -bfile data -p pheno.txt -farmcpu -o out

# Run multiple models at once
jx gwas -bfile data -p pheno.txt -lm -lmm -fastlmm -farmcpu -o out
```

提示：运行 `gwas` 时至少选择一个模型参数。

**Input**:

- `-vcf, --vcf` / `-bfile, --bfile` — 基因型来源（VCF 路径或 PLINK 前缀）。  
  默认：必填
- `-p, --pheno` — 表型文件（制表符分隔）。  
  默认：必填
- `-n, --ncol` — 表型列索引（0-based，可重复）。  
  默认：全部列

**Models**:

- `-lm, --lm` — 线性模型（LM）。  
  默认：`False`
- `-lmm, --lmm` — 线性混合模型（LMM）。  
  默认：`False`
- `-fastlmm, --fastlmm` — FaST-LMM 近似。  
  默认：`False`
- `-farmcpu, --farmcpu` — FarmCPU。  
  默认：`False`

**Relatedness &amp; Covariates**:

- `-k, --grm` — GRM 设置：`1`=居中，`2`=标准化，或预计算 GRM 路径。  
  默认：`1`
- `-q, --qcov` — 群体协变量：PC 数量或 Q 矩阵路径。  
  默认：`0`
- `-c, --cov` — 协变量文件（不含截距列）。  
  默认：`None`

**Variant QC**:

- `-maf, --maf` — 剔除 MAF 小于阈值的变异。  
  默认：`0.02`
- `-geno, --geno` — 剔除缺失率高于阈值的变异。  
  默认：`0.05`

**Output &amp; Performance**:

- `-plot, --plot` — 生成 Manhattan 与 QQ 图。  
  默认：`False`
- `-chunksize, --chunksize` — 流式处理的 SNP 块大小。  
  默认：`100000`
- `-mmap-limit, --mmap-limit` — BED 输入启用窗口 mmap（仅基准测试）。  
  默认：`False`
- `-t, --thread` — CPU 线程数（`-1`=全部）。  
  默认：`-1`
- `-o, --out` — 输出目录。  
  默认：`"."`
- `-prefix, --prefix` — 输出前缀（缺省自动生成）。  
  默认：auto

**Output files**:

| File | Description |
| ---- | ----------- |
| `{prefix}.{trait}.lm.tsv` | GLM 结果 |
| `{prefix}.{trait}.lmm.tsv` | LMM 结果 |
| `{prefix}.{trait}.fastlmm.tsv` | fastLMM 结果 |
| `{prefix}.{trait}.farmcpu.tsv` | FarmCPU 结果 |
| `{prefix}.{trait}.{model}.svg` | 直方、Manhattan 和 QQ 图（如启用 `--plot`） |

### Genomic Selection (`gs`)

运行 GBLUP、rrBLUP、BayesA/B/Cpi 等模型进行基因组预测。

```bash
# Run selected models
jx gs -vcf data.vcf.gz -p pheno.txt -GBLUP -rrBLUP -o out
jx gs -vcf data.vcf.gz -p pheno.txt -BayesA -BayesB -BayesCpi -o out

# Choose phenotype column and enable PCA-based dimensionality reduction and plotting with 5-fold cross-validazation.
jx gs -vcf data.vcf.gz -pheno pheno.txt -n 0 -GBLUP -pcd -plot -o out -cv 5
```

**Input**:

- `-vcf, --vcf` / `-bfile, --bfile` — 基因型来源（VCF 路径或 PLINK 前缀）。  
  默认：必填
- `-p, --pheno` — 表型文件（制表符分隔）。  
  默认：必填
- `-n, --ncol` — 表型列索引（0-based，可重复）。  
  默认：全部列

**Models**:

- `-GBLUP/--GBLUP`, `-rrBLUP/--rrBLUP`, `-BayesA/--BayesA`, `-BayesB/--BayesB`, `-BayesCpi/--BayesCpi` — 选择运行的模型。  
  默认：全部为 `False`

**Variant QC**:

- `-maf, --maf` — 剔除 MAF 小于阈值的变异。  
  默认：`0.02`
- `-geno, --geno` — 剔除缺失率高于阈值的变异。  
  默认：`0.05`

**Output &amp; Performance**:

- `-cv/--cv` — 启用 K 折交叉验证。  
  默认：`None`
- `-pcd/--pcd` — 启用 PCA 降维。  
  默认：`False`
- `-plot/--plot` — 生成预测与观测的散点图。  
  默认：`False`
- `-o/--out`, `-prefix/--prefix` — 输出目录/前缀。  
  默认：`"."` / auto

**Outputs**:

| File | Description |
| ---- | ----------- |
| `{prefix}.{trait}.gs.tsv` | 各模型的 GEBV 预测结果 |
| `{prefix}.{trait}.gs.{model}.svg` | 预测散点图（如启用 `--plot`） |

### postGWAS (`postGWAS`)

对 GWAS 结果进行可视化与注释（Manhattan、QQ、SNP 注释与高亮）。

```bash
# Basic plotting
jx postGWAS -f result.lmm.tsv

# Custom column names and threshold
jx postGWAS -f result.tsv -chr "chr" -pos "ps" -pvalue "p_wald" -threshold 1e-6

# Highlight regions and annotate significant SNPs
jx postGWAS -f result.tsv -hl snps.bed -a annotation.gff -ab 100
```

**Input**:

- `-f/--file` — GWAS 结果文件（支持 glob）。  
  默认：必填

**Name of columns**:

- `-chr/--chr`, `-pos/--pos`, `-pvalue/--pvalue` — 列名。  
  默认：`"chrom"`, `"pos"`, `"pwald"`

**Other parameters**:

- `-threshold/--threshold` — 显著性阈值。  
  默认：`0.05 / SNPs`
- `-noplot/--noplot` — 禁用绘图（仅注释）。  
  默认：`False`
- `-color/--color` — 配色方案（0-6）。  
  默认：`0`
- `-hl/--highlight` — 高亮区域 BED 文件。  
  默认：`None`
- `-format/--format` — 图像格式 `pdf/png/svg/tif`。  
  默认：`"png"`
- `-a/--anno` — 注释文件（GFF/GTF/BED）。  
  默认：`None`
- `-ab/--annobroaden` — 注释窗口（kb）。  
  默认：`None`
- `-descItem/--descItem` — GFF 描述字段键名。  
  默认：`"description"`
- `-o/--out`, `-prefix/--prefix`, `-t/--thread` — 输出目录/前缀；线程数（`-1`=全部）。  
  默认：`"."` / `"JanusX"` / `-1`

Highlight file format:

```text
chr1	start	end	name	description
chr1	1000000	1000000	GeneA	Description of GeneA
chr1	2000000	2000000	GeneB	Description of GeneB
```

Annotation file format (GFF example):

```text
chr1	.	gene	1000000	2000000	.	+	.	ID=gene1;description=Example gene
chr1	.	mRNA	1000000	2000000	.	+	.	ID=mRNA1;Parent=gene1
```

**Outputs**:

| File | Description |
| ---- | ----------- |
| `{prefix}.manh.{format}` | Manhattan 图 |
| `{prefix}.qq.{format}` | QQ 图 |
| `{prefix}.{threshold}.anno.tsv` | 注释后的显著 SNP（如启用） |

### Simulation (`sim`)

生成用于流程测试的模拟基因型/表型数据。

```bash
jx sim [nsnp(k)] [nidv] [outprefix]
```

## Citation

```bibtex
@article {FuJanusX,
  title = {JanusX: an integrated and high-performance platform for scalable genome-wide association studies and genomic selection},
  author = {Fu, Jingxian and Jia, Anqiang and Wang, Haiyang and Liu, Hai-Jun},
  year = {2026},
  doi = {10.64898/2026.01.20.700366},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2026/01/23/2026.01.20.700366},
  journal = {bioRxiv}
}
```
