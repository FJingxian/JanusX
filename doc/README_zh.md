# JanusX

[简体中文](./README_zh.md) | [English](../README.md) | [Zea Eureka](https://mp.weixin.qq.com/s/jl3h2DPRC21l8QJ0WxzXDA)

JanusX 是一个面向数量遗传分析的高性能工具集。项目采用 Rust（PyO3）加速核心计算，并提供统一的 Python CLI，用于 GWAS、GS、后处理可视化和变异处理流程。

## 项目概览

- 统一入口命令：`jx`
- GWAS 模型：`LM`、`LMM`、`fastLMM`、`FarmCPU`
- GS 模型：`GBLUP`、`rrBLUP`、`BayesA`、`BayesB`、`BayesCpi`
- 流式基因型读取（VCF/PLINK/TXT），支持低内存计算
- 集成后处理模块：`postgwas`、`postbsa`、`postgarfield`
- 其它模块：`grm`、`pca`、`gmerge`、`fastq2vcf`、`sim`、`simulation`

## 安装

环境要求：

- Python `>=3.9`
- Rust 工具链（仅在本地无可用 wheel、需要源码构建时）

### PyPI（推荐）

```bash
pip install -U janusx
jx -h
```

### GitHub 最新版

```bash
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/FJingxian/JanusX.git
jx -h
```

### Docker（源码构建镜像）

```bash
git clone https://github.com/FJingxian/JanusX.git
cd JanusX
docker build -t janusx:latest .
docker run --rm janusx:latest -h
```

### 已安装版本在线更新

```bash
jx --update
```

## CLI 模块

### Genome-wide Association Studies（GWAS）

- `grm`：基于基因型输入（`-vcf` 或 `-bfile`）构建亲缘关系矩阵
- `pca`：基于基因型、GRM 前缀或已有 PCA 前缀进行群体结构分析
- `gwas`：运行全基因组关联分析（`LM`、`LMM`、`fastLMM`、`FarmCPU`）
- `postgwas`：GWAS 结果后处理（Manhattan/QQ/注释/merge/LD 视图）

### Genomic Selection（GS）

- `gs`：基因组预测与模型选择

### GARFIELD

- `garfield`：基于随机森林的标记-性状关联分析
- `postgarfield`：GARFIELD 结果汇总与可视化

### Bulk Segregation Analysis（BSA）

- `postbsa`：BSA 结果后处理与可视化

### Pipeline and Utility

- `fastq2vcf`：从 FASTQ 到 VCF 的变异检测流程
- `gmerge`：多来源基因型/变异数据合并

### Benchmark

- `sim`：快速模拟流程
- `simulation`：扩展模拟与基准评估流程

查看完整参数：

```bash
jx <module> -h
```

## 快速开始

### 1) GWAS

```bash
jx gwas -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -lmm -plot -o test
```

### 2) postgwas（单文件）

```bash
jx postgwas -gwasfile test/mouse_hs1940.test0.lmm.tsv -manh -qq -threshold 1e-6 -o testpost
```

### 3) postgwas merge（多个 GWAS 结果绘制到一张图）

```bash
jx postgwas \
  -gwasfile testb/mouse_hs1940.test0.add.lmm.tsv \
  -merge testb/mouse_hs1940.test0.dom.lmm.tsv \
  -merge testb/mouse_hs1940.test0.het.lmm.tsv \
  -merge testb/mouse_hs1940.test0.rec.lmm.tsv \
  -manh -qq -o testpost
```

### 4) GS

```bash
jx gs -vcf example/mouse_hs1940.vcf.gz -p example/mouse_hs1940.pheno -GBLUP -cv 5 -plot -o testgs
```

### 5) postbsa

```bash
jx postbsa -file QBproject/5.table/all.tsv -b1 JS -b2 JR -window 1 -format pdf -o QBproject/5.table
```

### 6) FASTQ 到 VCF

```bash
jx fastq2vcf -r /path/ref.fa -i /path/fastq_dir -w /path/workdir -b nohup -m 2
```

## 输入文件格式

### 表型文件

使用制表符分隔。第一列为样本 ID，后续列为表型。

```text
sample	trait1	trait2
id1	10.5	0.85
id2	12.3	1.00
id3	8.6	0.92
```

### 基因型文件

- VCF：`.vcf` 或 `.vcf.gz`（参数 `-vcf`）
- PLINK：`.bed/.bim/.fam` 前缀（参数 `-bfile`）
- 文本基因型矩阵（部分模块支持 `-file`）：
  - 第一行为样本 ID
  - 后续每行为 SNP 主序的数值型基因型

文本矩阵示例：

```text
id1	id2	id3	id4
0	1	2	0
2	2	1	0
0	0	1	1
```

### GWAS 可选矩阵输入

- GRM（`gwas` 中 `-k/--grm`）：与基因型样本顺序一致的数值矩阵
- Q/PC 协变量（`gwas` 中 `-q/--qcov`）：PC 数量或协变量矩阵文件
- 其它协变量（`gwas` 中 `-c/--cov`）：与基因型样本顺序一致的数值矩阵

以上矩阵建议为纯数值、空格/制表符分隔、无行列名。

### GRM/PCA 输入约定（已更新）

- `jx grm` 仅接受基因型输入：`-vcf` 或 `-bfile`
- `jx grm` 输出：
  - 默认文本：`{prefix}.grm.txt` 与 `{prefix}.grm.txt.id`
  - `--npy` 二进制：`{prefix}.grm.npy` 与 `{prefix}.grm.npy.id`
- `jx pca` 输入（互斥）：
  - 基因型输入：`-vcf` 或 `-bfile`（从基因型计算 PCA）
  - GRM 前缀：`-k/--grm <prefix>`（要求存在 `{prefix}.grm.id` 以及 `{prefix}.grm.txt` 或 `{prefix}.grm.npy`）
  - 已有 PCA 前缀：`-q/--qcov <prefix>`（要求存在 `{prefix}.eigenval` 与 `{prefix}.eigenvec`，仅可视化模式）

## 常见输出命名

- `gwas`：`{prefix}.{trait}.{model}.tsv`，可选图文件 `{prefix}.{trait}.{genetic_model}.{model}.svg`
- `postgwas`：`{prefix}.manh.{format}`、`{prefix}.qq.{format}`、`{prefix}.{thr}.anno.tsv`
- `postgwas` merge：`{prefix}.merge.manh.{format}`、`{prefix}.merge.qq.{format}`
- `postbsa`：
  - `{prefix}.{bulk1}vs{bulk2}.snpidx.{format}`
  - `{prefix}.{bulk1}vs{bulk2}.bsa.{format}`
  - `{prefix}.{bulk1}vs{bulk2}.smooth.tsv`
  - `{prefix}.{bulk1}vs{bulk2}.thr.tsv`
- `gs`：`{prefix}.{trait}.gs.tsv`

## Python API（示例）

```python
from janusx.gfreader import load_genotype_chunks

for geno, sites in load_genotype_chunks("example/mouse_hs1940.vcf.gz", chunk_size=50000):
    # geno: (m, n) float32 基因型块
    # sites: 位点元信息列表
    pass
```

```python
from janusx.gtools.wgcna import cor, adj, tom, cluster
```

## 引用

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

## 许可证

MIT License，见 [LICENSE](../LICENSE)。
