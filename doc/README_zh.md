# JanusX：高效的全基因组关联分析及全基因组选择工具

[English](../README.md) | [简体中文(推荐)](./README_zh.md)

## 引言

&emsp;&emsp;随着基因组学研究的深入，海量基因型和表型数据的分析需求日益增长。现有全基因组关联分析(GWAS)软件如GEMMA、GCTA、rMVP和TASSEL等在处理大规模数据时存在一些局限性：GEMMA缺乏Windows版本，跨平台兼容性不足；多核并行计算效率低下；计算亲缘关系矩阵(Q矩阵)依赖外部工具等
&emsp;&emsp;为解决这些问题，我们基于混合线性模型算法进行了深度优化，开发了JanusX工具。该工具在计算效率、跨平台兼容性和易用性方面均有显著提升。源代码已发布[Github仓库](https://github.com/MaizeMan-JxFU/JanusX)，欢迎进行测试使用。

## 算法原理

### 混合线性模型

&emsp;&emsp;GWAS中的混合线性模型如公式(1)所示，通常简写成公式(2)的形式。其中$X$是固定因子矩阵，包含1列全为1的固定截距向量以及多列固定因子向量(例如 群体结构、基因型)，$\beta$是固定因子的效应值；$g$是个体随机因子，$g \sim N(0,\sigma_{g}^2 G)$，$G$是亲缘关系矩阵，可通过系谱关系或基因型计算获得；$\epsilon$ 是残差，$\epsilon \sim N(0,\sigma_{\epsilon}^2 I)$ ；y是表型向量，$y \sim N(X\beta,\sigma_{g}^2 G+\sigma_{\epsilon}^2 I)$

```math
y=\mu+X_{cov}\beta_{cov}+X_{snp}\beta_{snp}+g+\epsilon \tag{1}
```

```math
y=X\beta+g+\epsilon \tag{2}
```

&emsp;&emsp;其中需要估计的参数包括$\beta、\sigma_{g}^2、\sigma_{\epsilon}^2$。

&emsp;&emsp;首先，我们设：$\sigma_{\epsilon}^2=\lambda \sigma_{g}^2$, $V=G+\lambda I $ , 则 $y\sim N(X\beta,\sigma_{g}^2 V)$

&emsp;&emsp;基于广义最小二乘法，我们可以用 $V$ 估计 $\beta$ ,公式如下：

$$\hat{\beta}=(X'V^{-1}X)^{-1}X'V^{-1}y \tag{3}$$

>**广义最小二乘法的推导**
*定理1* 任意正定矩阵A都存在$A=LL'$(Cholesky分解)
当 $y=X\beta+\epsilon$ ，$\epsilon \sim N(0,\sigma^{2} I)$ ，其中 $I$ 是单位矩阵。易证明 $\beta$ 的最小二乘估计公式 $\hat{\beta}=(X'X)^{-1}X'y$
我们可以把混合线性模型同样视为 $y=X\beta+\epsilon$ 的形式，但此时 $\epsilon \sim N(0,\sigma^{2} \Sigma)$ ，其中$\Sigma$是随机因子的协方差矩阵，那么我们只需要对方程进行线性变换将 $\Sigma$ 转变为单位矩阵则可以套用最小二乘估计的公式。根据定理1，可以将正定矩阵$\Sigma$分解成可逆的上下三角矩阵 $L$ 和 $L'$ ，随后对混合线性模型公式同乘 $L^{-1}$ 进行线性变换，即可将 $\Sigma$ 转换为单位矩阵。推导如下：
$$
\Sigma=LL' \\
L^{-1}y=L^{-1}X\beta+L^{-1}\epsilon, L^{-1}\epsilon \sim N(0,\sigma^2 I) \\
\hat{\beta}=((L^{-1}X)'(L^{-1}X))^{-1}(L^{-1}X)'L^{-1}y \\
=(X'(L^{-1})'L^{-1}X)^{-1}X'(L^{-1})'L^{-1}y \\
\because (L^{-1})'L^{-1}=\Sigma^{-1},\therefore \hat{\beta}=(X'\Sigma^{-1}X)^{-1}X'\Sigma^{-1}y
$$

&emsp;&emsp;此时，需要估计的参数包括 $\sigma_{g}^2、\lambda$ ，我们采用限制性最大似然法对其进行估计，或者说我们将表型值向量$y$的多元正态分布的似然函数作为损失函数估计这两个未知参数。多元正态分布的限制性似然函数公式([推导](https://xiuming.info/docs/tutorials/reml.pdf)较为复杂，直接上公式)如下：

设

$$r=(y-X\beta)'(y-X\beta) \tag{4}$$

则

```math
\ln{L_{ml}}=-\frac{1}{2}\ln{\lvert \sigma_{g}^{2}V \rvert}-\frac{1}{2}\sigma_{g}^{-2}r'V^{-1}r-\frac{N}{2}\ln{2\pi} \\
\ln{L_{reml}}=\ln{L_{ml}}-\frac{1}{2}\ln{\lvert \sigma_{g}^{-2}X'V^{-1}X \rvert} \tag{5}
```

令 $\frac{\partial \ln{L_{reml}}}{\partial \sigma}=0$，解得：

$$\hat{\sigma_{g}^2}=\frac{r'V^{-1}r}{n-p} \tag{6}$$

将公式3代入公式4，公式4代入公式5和公式6，公式6代入公式5，化简得到：

```math
C_{reml}=\frac{n-p}{2}(\ln{(n-p)}-\ln{2\pi}-1) \\
\ln{L_{reml}}=C_{reml}-(n-p)\ln{(r'V^{-1}r)}+\ln{\lvert V\rvert}+\ln{\lvert X'V^{-1}X \rvert} \tag{7}
```

&emsp;&emsp;至此，我们的目标是最大化公式7所示的对数似然函数 $lnL_{reml}$ 。而我们惊奇地发现，只需要迭代估计一个参数 $\lambda$ ，剩下的其他参数都可以由估计的 $\lambda$ 求解出来(公式3和公式6)。这种情况下，我们可以不用牛顿法而求解极复杂的公式7的一阶导(JacbiMatrix)和二阶导(HessianMatrix)，可以直接对公式7采用布伦特法(brent)搜索 $LL(\lambda)$ 函数最大值对应的 $\lambda$ 。

&emsp;&emsp;我们发现每次迭代都需要对协方差矩阵V进行求逆计算，这是极为消耗计算资源的，尤其是上千万SNP位点进行运算的时候。那么有没有方法简化协方差矩阵的求逆呢？当然有的，前人给出的解决方案就是对G矩阵进行奇异值分解(SVD)，随后在对公式2进行线性变换，将G矩阵转化为对角矩阵，这样协方差矩阵也就转变成了对角矩阵。对角矩阵的求逆和求行列式都极为简单，极大简化了之前的计算复杂。SVD不愧是线性代数的一大工具，无论是经典的混合线性模型，还是现在火热的深度学习，都占据着举足轻重的地位。

>**线性变换简化公式2协方差矩阵的推导**
*定理1* 任意矩阵都可以通过奇异值分解成左奇异矩阵(方阵)、奇异值矩阵(对角矩阵)和右奇异矩阵(方阵)。其中左右奇异矩阵都是共轭矩阵，$A=USV'$
首先将G矩阵进行SVD分解，分解成 $U$ 、$S$ 和 $U'$三个矩阵，再对公式2左右两边同乘共轭矩阵 $U'$ 即可简化V为对角矩阵
$$
G=USU' \\
U'y=U'X\beta+U'g+U'\epsilon \\
U'y\sim N(U'X\beta,\sigma_{g}^{2}U'GU+\sigma_{\epsilon}^{2}U'U ) \\
\because U'U=I, \therefore U'y\sim N(U'X\beta,\sigma_{g}^{2}S+\sigma_{\epsilon}^{2}I ) \\
\therefore V=S+\lambda I
$$

### 主成分求解优化

&emsp;&emsp;样本-基因型矩阵的主成分通常作为群体结构加入固定效应。传统SVD分解计算复杂度为$O(n^3)$，对于高维矩阵效率低下。随机奇异值分解(Random SVD)通过随机投影和子空间迭代，可高效计算前$k$个主成分，显著降低计算负担。

## GWAS测试

**对比软件**：[GEMMA](https://github.com/genetics-statistics/GEMMA)、GCTA以及rMVP
**测试平台**：Ubuntu 22.04.5 LTS(x86_64), 2*Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz
**测试数据集**: [RiceAtlas](http://60.30.67.242:18076/#/download)，6048个体 Panicle_length性状 (基因型+表型)
**数据格式**：
基因型属于plink标准格式
表型格式如下，每列是不同样本，第二列~第n列是多种表型。换行符作为分隔。

| samples | pheno_name |
| :-----: | :------: |
| indv1 | phenovalue 1 |
| indv2 | phenovalue 2 |
| ... | ... |
| indvn | phenovalue n |

**测试代码**：
GEMMA:

```bash
# 表型预处理 去表头 创建双id
awk -F "\t" {'print $1,$1,$2'} ~/data_pub/1.database/RiceAtlas/1.pheno.blup.tsv | tail +2 > data/test.pheno
python -c "import pandas as pd;data = pd.read_csv('test.pheno',sep='\s+',header=None);data.fillna('-9').to_csv('test.pheno.gemma.gcta.txt',sep=' ',index=None,header=None)"
plink --bfile ~/data_pub/1.database/RiceAtlas/Rice6048 --pheno data/test.pheno.gemma.gcta.txt --make-bed --out data/test
# 测试
time gemma -bfile data/test -gk -o gemma
# GEMMA 0.98.5 (2021-08-25) by Xiang Zhou, Pjotr Prins and team (C) 2012-2021
# Reading Files ... 
# ## number of total individuals = 6048
# ## number of analyzed individuals = 3381
# ## number of covariates = 1
# ## number of phenotypes = 1
# ## number of total SNPs/var        =  5694922
# ## number of analyzed SNPs         =  4828877
# Calculating Relatedness Matrix ... 
# ================================================== 100%
# **** INFO: Done.

# real    30m38.194s
# user    1432m23.940s
# sys     112m30.275s
time ./gemma -bfile data/test -k output/gemma.cXX.txt -lmm -o gemma
# GEMMA 0.98.5 (2021-08-25) by Xiang Zhou, Pjotr Prins and team (C) 2012-2021
# Reading Files ... 
# ## number of total individuals = 6048
# ## number of analyzed individuals = 3381
# ## number of covariates = 1
# ## number of phenotypes = 1
# ## number of total SNPs/var        =  5694922
# ## number of analyzed SNPs         =  4828877
# Start Eigen-Decomposition...
# pve estimate =0.91255
# se(pve) =0.00754577
# ================================================== 100%
# **** INFO: Done.

# real    192m26.571s
# user    627m6.371s
# sys     60m49.919s
```

GCTA:

```bash
awk -F "\t" {'print $1,$1,$2'} ~/data_pub/1.database/RiceAtlas/1.pheno.blup.tsv | tail +2 > data/test.pheno
# GCTA 支持多线程 --thread-num 92
time gcta64 --bfile data/test --autosome --make-grm 1 --out gcta  --thread-num 92
# *******************************************************************
# * Genome-wide Complex Trait Analysis (GCTA)
# * version v1.94.1 Linux
# * Built at Nov 15 2022 21:14:25, by GCC 8.5
# * (C) 2010-present, Yang Lab, Westlake University
# * Please report bugs to Jian Yang <jian.yang@westlake.edu.cn>
# *******************************************************************
# Analysis started at 15:55:21 CST on Thu Oct 30 2025.
# Hostname: user-NF5466M6

# Options: 
 
# --bfile data/test 
# --autosome 
# --make-grm 1 
# --out gcta 
# --thread-num 92 

# The program will be running with up to 92 threads.
# Note: GRM is computed using the SNPs on the autosomes.
# Reading PLINK FAM file from [data/test.fam]...
# 6048 individuals to be included from FAM file.
# 6048 individuals to be included. 0 males, 0 females, 6048 unknown.
# Reading PLINK BIM file from [data/test.bim]...
# 5694922 SNPs to be included from BIM file(s).
# Computing the genetic relationship matrix (GRM) v2 ...
# Subset 1/1, no. subject 1-6048
#   6048 samples, 5694922 markers, 18292176 GRM elements
# IDs for the GRM file have been saved in the file [gcta.grm.id]
# Computing GRM...
#   23.0% Estimated time remaining 16.8 min
#   65.3% Estimated time remaining 5.3 min
#   100% finished in 789.4 sec
# 5694922 SNPs have been processed.
#   Used 5694922 valid SNPs.
# The GRM computation is completed.
# Saving GRM...
# GRM has been saved in the file [gcta.grm.bin]
# Number of SNPs in each pair of individuals has been saved in the file [gcta.grm.N.bin]

# Analysis finished at 16:08:38 CST on Thu Oct 30 2025
# Overall computational time: 13 minutes 16 sec.

# real    13m16.721s
# user    994m32.127s
# sys     6m53.113s
time gcta64 --bfile data/test --pheno data/test.pheno --grm gcta --mlma --out gcta  --thread-num 92
# *******************************************************************
# * Genome-wide Complex Trait Analysis (GCTA)
# * version v1.94.1 Linux
# * Built at Nov 15 2022 21:14:25, by GCC 8.5
# * (C) 2010-present, Yang Lab, Westlake University
# * Please report bugs to Jian Yang <jian.yang@westlake.edu.cn>
# *******************************************************************
# Analysis started at 16:19:06 CST on Thu Oct 30 2025.
# Hostname: user-NF5466M6

# Accepted options:
# --bfile data/test
# --pheno data/test.pheno
# --grm gcta
# --mlma 
# --out gcta
# --thread-num 92

# Note: the program will be running on 92 threads.

# Reading PLINK FAM file from [data/test.fam].
# 6048 individuals to be included from [data/test.fam].
# Reading PLINK BIM file from [data/test.bim].
# 5694922 SNPs to be included from [data/test.bim].
# Reading PLINK BED file from [data/test.bed] in SNP-major format ...
# Genotype data for 6048 individuals and 5694922 SNPs to be included from [data/test.bed].
# Reading phenotypes from [data/test.pheno].
# Non-missing phenotypes of 3487 individuals are included from [data/test.pheno].
# Reading IDs of the GRM from [gcta.grm.id].
# 6048 IDs are read from [gcta.grm.id].
# Reading the GRM from [gcta.grm.bin].
# GRM for 6048 individuals are included from [gcta.grm.bin].
# 3487 individuals are in common in these files.

# Performing MLM association analyses (including the candidate SNP) ...

# Performing  REML analysis ... (Note: may take hours depending on sample size).
# 3487 observations, 1 fixed effect(s), and 2 variance component(s)(including residual variance).
# Calculating prior values of variance components by EM-REML ...
# Updated prior values: 121.369 128.583
# logL: -11046.3
# Running AI-REML algorithm ...
# Iter.   logL    V(G)    V(e)
# 1       -11028.78       127.62465       129.37499
# 2       -11026.69       132.87349       129.65568
# 3       -11025.57       137.22864       129.61548
# 4       -11024.97       140.81534       129.38948
# 5       -11024.63       150.11953       128.37421
# 6       -11024.20       153.88104       127.03504
# 7       -11024.14       155.59620       126.40843
# 8       -11024.13       156.37140       126.12474
# 9       -11024.13       156.71952       125.99722
# 10      -11024.13       156.87542       125.94010
# 11      -11024.13       156.94514       125.91454
# 12      -11024.13       156.97631       125.90312
# 13      -11024.13       156.99023       125.89801
# Log-likelihood ratio converged.
# Calculating allele frequencies ...

# Running association tests for 5694922 SNPs ...
# ^C

# real    978m5.950s
# user    999m5.255s
# sys     60m27.973s
```

JanusX:

```bash
# 表型预处理
awk -F "\t" -v OFS="\t" {'print $1,$2'} ~/data_pub/1.database/RiceAtlas/1.pheno.blup.tsv > data/test.pheno
# 默认开启所有线程 保持和GCTA一致 使用 --thread 92. 和其他方法保持一致不使用q矩阵
jx gwas --bfile data/test --pheno data/test.pheno --out . --thread 92 --qcov 0 --grm gemma1
# High Performance Linear Mixed Model Solver for Genome-Wide Association Studies
# Host: user-NF5466M6

# ************************************************************
# GWAS LMM SOLVER CONFIGURATION
# ************************************************************
# Genotype file:    data/test
# Phenotype file:   data/test.pheno
# Output directory: output
# GRM method:       VanRanden
# Q matrix:         0
# Covariant matrix: None
# Threads:          92 (User specified)
# HighAC mode:      True
# ************************************************************

# Loading genotype from data/test.bed...
# Loading phenotype from data/test.pheno...
# Geno and Pheno are ready!
# * Loading GRM from data/test.k.VanRanden.txt...
# GRM (6048, 6048):
# [[1.370728 0.887475 0.888342 0.894397 0.83152 ]
#  [0.887475 1.332102 1.050703 0.941735 0.975881]
#  [0.888342 1.050703 1.323239 1.011246 1.140967]
#  [0.894397 0.941735 1.011246 1.344044 0.999469]
#  [0.83152  0.975881 1.140967 0.999469 1.350127]]
# Qmatrix (6048, 0):
# []
# ************************************************************
# Phenotype: Panicle_length, Number of samples: 3381, Number of SNP: 5694922, pve of null: 0.918, high AC model: True
# CPU: 38.4%, Memory: 54.42 G, Process: 100.0% (time cost: 13.25/13.25 mins)
# Effective number of SNP: 4341840
# Saved in output/Panicle_length.assoc.tsv
# Visualizing...
# Saved in output/Panicle_length.png

# Time costed: 1035.31 secs


# Finished, Total time: 1035.31 secs
# 2025-10-31 16:45:13
```

### 准确性测试

&emsp;&emsp;对上述比较测试中JanusX和GEMMA生成的结果进行比较。结果显示，在使用各自计算的亲缘关系矩阵下，即使算法设计一致，也可能由于每一步计算误差的级联反应而对结果产生一定偏差（如下第一张图）；在统一使用GEMMA计算的亲缘关系矩阵的情况下，结果近乎完全一致（如下第二张图）。

![ACtest2](../fig/ac_test2.png "Test of accuracy compared with GEMMA"){: style="display: block; margin: 0 auto"}

![ACtest](../fig/ac_test.png "Test of accuracy compared with GEMMA"){: style="display: block; margin: 0 auto"}

### 效率测试

&emsp;&emsp;为了测试代码多线程的效率，本研究在服务器上固定基因型、表型，探究CPU核心数和耗时之间的关系。理想的并行计算情况下，耗时应该会随着核心数的增加而线性降低。然而，结果表明耗时随CPU核心的增加呈现负指数衰减，并非线性关系。最高效的计算方法是评估基因型读取时间和单个表型计算时间，最终确定拆分计算亦或合并计算。(CUBIC群体，476个体，9.8M SNP，6个锈病相关表型)

|Core|Time (min)|Memory (gb)|
|:-:|:-:|:-:|
|8|69|8.66|
|16|42|9.42|
|32|28|11.04|
|64|23|14.96|

![CPUtest](../fig/cpu_time_test.png "Test of relation between CPU and costed time"){: style="display: block; margin: 0 auto"}

### 结论

&emsp;&emsp;总体而言，使用统一模型的前提下计算结果一致，JanusX相较于其他程序调用多线程更加积极、计算速度更快。
&emsp;&emsp;那么JanusX是完美的吗？当然不是，他的缺点是内存占用显著高于现有的软件，$M_{1,000 \times 1,000,000}$的矩阵内存占用高达12g，是其他软件的内存占用的2-3倍。此外随着个体数的增加，初始化时间（主要是奇异值分解的时间）将会显著增长。而杨剑等的[fast-gwa](https://doi.org/10.1038/s41588-019-0530-8)在2019年就实现了算法改进，使其GWAS模型适用于百万级个体的关联分析。

## 使用方法

### 安装

首先需要环境中需要包含 [python](https://www.python.org/downloads/release/python-3139/) (3.9~3.13)
如果有git基础，以下几行代码即可完成安装啦~

```bash
# 网络顺畅的情况
# git clone https://github.com/MaizeMan-JxFU/JanusX.git
# 不能科学上网可以选择国内代理
git clone https://gh-proxy.com/https://github.com/MaizeMan-JxFU/JanusX.git
# 进入目标文件夹
cd JanusX
sh ./install.sh

```

**注意**: Windows安装已不再支持。请使用Linux/macOS或Windows子系统Linux (WSL)。

### Modules

|模块|功能|最简语法|
|-|-|-|
|**gwas**|利用混合线性模型进行全基因组关联分析|```jx gwas -h```|
|**gs**|利用BLUP和机器学习模型进行全基因组选择|```jx gs -h```|
|**postGWAS**|可视化及注释全基因组关联分析的结果|```jx postGWAS -h```|
|**pca**|谱分解计算输入文件行主成分及可视化|```jx pca -h```|
|**grm**|亲缘关系矩阵估计(1-VanRanden;2-Yang)|```jx grm -h```|

|计划更新模块|功能|最简语法|
|-|-|-|
|**gwas: FarmCPU**|利用REM和FEM进行全基因组关联分析|```jx gwas --farmcpu ...```|
|**iqtree** (extModule)|最大似然法计算输入样本的系统发育关系|```jx iqtree ...```|
|...|...|...|

### 功能1: 全基因组关联分析

模块: gwas
必须参数1：--vcf [vcf文件] 或 --bfile [plink文件]
必须参数2：--pheno [表型文件]
必须参数3：--out [结果文件输出文件夹(不存在则自动创建)]
默认参数：计算 VanRanden 亲缘关系矩阵、基于基因型的前3个主成分，生成文件于vcf或bfile文件目录

```bash
jx <module> [options]

# 查看帮助
jx gwas --help

# 混合线性模型
jx gwas --vcf example/mouse_hs1940.vcf.gz --pheno example/mouse_hs1940.pheno --out test  # vcf format
jx gwas --npy example/mouse_hs1940 --pheno example/mouse_hs1940.pheno --out test  # numpy format

# 可视化
jx postGWAS --files test/test0.assoc.tsv --threshold 1e-6
```

使用example文件夹下的[测试数据](https://doi.org/10.1038/ng.3609)输出结果如下所示：
![GWAStest](../fig/test0.png "GWAS test of JanusX"){: style="display: block; margin: 0 auto"}
*(The above image depicts physiological and behavioral trait loci identified in CFW mice using GEMMA, from Parker et al, Nature Genetics, 2016.)

### 功能2: 全基因组选择

xxx

## 写在最后

更新计划：重写bed_reader函数，获得更高效的基因型编码方式，从而降低内存占用；优化biokitplot中的GWASplot函数，使其绘制千万级别位点的速度更快、内存占用更低。混合线性模型中加入牛顿法，解决多随机效应方差估计的问题（目前JanusX只能引入单一随机效应，例如亲缘关系）。
更多用法可以访问[Github仓库](https://github.com/MaizeMan-JxFU/JanusX)，仍在更新中...
