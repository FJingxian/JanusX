# JanusX CLI 文档

本文档描述 JanusX 所有命令行工具的用法。

---

## 目录 | Table of Contents

1. [快速开始 | Quick Start](#快速开始--quick-start)
2. [GWAS 全基因组关联分析](#gwas-全基因组关联分析)
3. [GS 基因组选择](#gs-基因组选择)
4. [GRM 亲缘关系矩阵](#grm-亲缘关系矩阵)
5. [PCA 主成分分析](#pca-主成分分析)
6. [postGWAS 结果可视化](#postgwas-结果可视化)
7. [SIM 数据模拟](#sim-数据模拟)

---

## 快速开始

```bash
# 查看帮助
jx -h
jx <module> -h

# 示例：运行 GWAS 分析
jx gwas -vcf example.vcf.gz -p phenotype.txt -lmm -o results
```

### 输入格式

**基因型文件 (Genotype)**:

- VCF: `.vcf` 或 `.vcf.gz`
- PLINK: `.bed`/`.bim`/`.fam` (使用前缀)

**表型文件 (Phenotype)**:

- 制表符分隔的文本文件
- 第一列：样本ID
- 后续列：表型值

| samples | trait1 | trait2 |
|---------|--------|--------|
| indv1   | 10.5   | 0.85   |
| indv2   | 12.3   | 0.92   |

**GWAS 可选矩阵示例（GRM / Q / 协变量）**:

GRM（亲缘矩阵）示例（N x N，行/列顺序需与基因型样本顺序一致）：

```text
1.00   0.12   0.05
0.12   1.00   0.08
0.05   0.08   1.00
```

Q（PC）矩阵示例（N x K，行顺序需与基因型样本顺序一致）：

```text
0.12   -0.03
-0.05  0.08
0.02   -0.01
```

协变量矩阵示例（N x C，行顺序需与基因型样本顺序一致）：

```text
0    1
1    1
0    2
```

注意：传给 `--grm`/`-k`、`--qcov`/`-q`、`--cov`/`-c` 的文件需为纯数值矩阵，
不包含样本 ID 或列名，用空格或 `\t` 分隔，且行顺序必须与基因型样本顺序一致。

---

## GWAS 全基因组关联分析

### 模块说明

全基因组关联分析 (Genome-Wide Association Study, GWAS) 用于识别与性状关联的遗传变异位点。JanusX 支持三种统计模型，可根据数据特征选择合适的分析方法。

| 模型 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **GLM** | 大样本、无明显群体结构 | 速度快、内存占用低 | 无法控制群体结构导致的假阳性 |
| **LMM** | 有群体结构或亲缘关系 | 控制假阳性、检验力较高 | 计算量较大 |
| **fastLMM** | 固定 lambda 的混合模型 | 速度快、适合筛选 | 近似解 |
| **FarmCPU** | 需要平衡检验力和假阳性 | 双重检验、控制混杂 | 需要全量基因型，内存要求高 |

### 使用方法

```bash
# GLM 模型 (流式/低内存)
jx gwas -vcf data.vcf.gz -p pheno.txt -lm -o out/

# LMM 模型 (流式/低内存)
jx gwas -vcf data.vcf.gz -p pheno.txt -lmm -o out/

# fastLMM 模型 (固定 lambda)
jx gwas -vcf data.vcf.gz -p pheno.txt -fastlmm -o out/

# FarmCPU 模型 (全内存)
jx gwas -vcf data.vcf.gz -p pheno.txt -farmcpu -o out/

# 同时运行多个模型
jx gwas -vcf data.vcf.gz -p pheno.txt -lm -lmm -fastlmm -farmcpu -o out/

# 使用预计算 GRM / Q(PC) / 协变量矩阵
jx gwas -vcf data.vcf.gz -p pheno.txt -lmm -k path/to/grm.txt -q path/to/q.txt -c cov.txt -o out/
```

说明：
- GRM 文件：数值型 N x N 矩阵，行列顺序需与基因型样本顺序一致。
- Q(PC) 文件：数值型 N x K 矩阵，行顺序需与基因型样本顺序一致。
- 协变量文件：数值型 N x C 矩阵，行顺序需与基因型样本顺序一致。
- 三者均为纯数值矩阵，不包含样本 ID 或列名，空格或 `\t` 分隔。

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-vcf/--vcf` | VCF 基因型文件路径 | 必填 |
| `-bfile/--bfile` | PLINK 二进制格式前缀 | 必填 |
| `-p/--pheno` | 表型文件路径 | 必填 |
| `-n/--ncol` | 表型列索引 (1-based，可多次指定) | 全部列 |
| `-lm/--lm` | 运行 GLM 模型 | False |
| `-lmm/--lmm` | 运行 LMM 模型 | False |
| `-fastlmm/--fastlmm` | 运行 fastLMM (固定 lambda) | False |
| `-farmcpu/--farmcpu` | 运行 FarmCPU 模型 | False |
| `-k/--grm` | GRM 计算方法: `1`(居中), `2`(标准化), 或预计算矩阵路径 | "1" |
| `-q/--qcov` | 主成分数量或 Q 矩阵文件路径 | "0" |
| `-c/--cov` | 协变量文件路径 (不含截距) | None |
| `-plot/--plot` | 生成 Manhattan 图和 QQ 图 | False |
| `-chunksize/--chunksize` | 流式处理的 SNP 块大小 | 100000 |
| `-t/--thread` | CPU 线程数 (`-1` 表示全部可用核心) | -1 |
| `-o/--out` | 输出目录路径 | "." |
| `-prefix/--prefix` | 输出文件前缀 | 自动生成 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.{trait}.lm.tsv` | GLM 模型结果 |
| `{prefix}.{trait}.lmm.tsv` | LMM 模型结果 |
| `{prefix}.{trait}.fastlmm.tsv` | fastLMM 模型结果 |
| `{prefix}.{trait}.farmcpu.tsv` | FarmCPU 模型结果 |
| `{prefix}.{trait}.{model}.svg` | 直方、Manhattan 和 QQ 图 (如启用) |

**结果文件格式** (制表符分隔):

| 列名 | 说明 |
|------|------|
| #CHROM | 染色体编号 |
| POS | 物理位置 (bp) |
| REF | 参考等位基因 |
| ALT | 替代等位基因 |
| maf | 次等位基因频率 |
| beta | 效应值 |
| se | 标准误 |
| p | P 值 |

---

## GS 基因组选择

### 模块说明

基因组选择 (Genomic Selection, GS) 利用全基因组标记信息预测个体的育种值 (GEBV)，适用于动植物育种早期选择。

**支持的模型**:

| 模型 | 方法类型 | 特点 |
|------|----------|------|
| **GBLUP** | 线性模型 | 基于亲缘关系矩阵，计算稳定 |
| **rrBLUP** | 岭回归 | 无需亲缘关系，适合高维数据 |
| **BayesA** | 贝叶斯模型 | 标记效应为 scaled-t 先验，适合多基因性状 |
| **BayesB** | 贝叶斯模型 | 变异选择，标记方差独立 |
| **BayesCpi** | 贝叶斯模型 | 变异选择，共享标记方差 |

### 使用方法

```bash
# 运行全部模型
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --rrBLUP -o out/

# 指定模型
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP -o out/

# 运行 Bayes 系列模型
jx gs --vcf data.vcf.gz --pheno pheno.txt --BayesA --BayesB --BayesCpi -o out/

# 指定特定表型列 (1-based 索引)
jx gs --vcf data.vcf.gz --pheno pheno.txt --n 0 --GBLUP -o out/

# 启用 PCA 降维并生成可视化
jx gs --vcf data.vcf.gz --pheno pheno.txt --GBLUP --pcd --plot -o out/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-vcf/--vcf` | VCF 基因型文件路径 | 互斥组 |
| `-bfile/--bfile` | PLINK 格式前缀 | 互斥组 |
| `-p/--pheno` | 表型文件路径 | 必填 |
| `-GBLUP/--GBLUP` | 使用 GBLUP 模型 | False |
| `-rrBLUP/--rrBLUP` | 使用 rrBLUP 模型 | False |
| `-BayesA/--BayesA` | 使用 BayesA 模型 | False |
| `-BayesB/--BayesB` | 使用 BayesB 模型 | False |
| `-BayesCpi/--BayesCpi` | 使用 BayesCpi 模型 | False |
| `-pcd/--pcd` | 启用 PCA 降维 (自动选择主成分数) | False |
| `-n/--ncol` | 表型列索引 (1-based) | 全部列 |
| `-plot/--plot` | 生成预测结果散点图 | False |
| `-o/--out` | 输出目录 | 当前目录 |
| `-prefix/--prefix` | 输出文件前缀 | 自动生成 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.{trait}.gs.tsv` | 各模型的 GEBV 预测值 |
| `{prefix}.{trait}.gs.{model}.svg` | 预测值与真实值散点图 (如启用) |

**结果文件格式** (制表符分隔):

| 列名 | 说明 |
|------|------|
| indv | 个体 ID |
| method1 | 模型1的预测值 |
| method2 | 模型2的预测值 |
| ... | 其他模型预测值 |

---

## GRM 亲缘关系矩阵

### 模块说明

遗传关系矩阵 (Genomic Relationship Matrix, GRM) 描述个体间的遗传相似度，是 LMM 模型和 GBLUP 的核心组成部分。

**计算方法**:

| 方法 | 公式类型 | 适用场景 |
|------|----------|----------|
| **方法1** (VanRaden) | 中心化 GRM | 默认方法，适用于大多数情况 |
| **方法2** (Yang) | 标准化 GRM | 需要考虑等位基因频率差异时 |

### 使用方法

```bash
# 默认方法 (居中 GRM)
jx grm --vcf data.vcf.gz -o out/

# 标准化方法
jx grm --vcf data.vcf.gz --method 2 -o out/

# 输出为压缩格式 (节省空间)
jx grm --vcf data.vcf.gz --npz -o out/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-vcf/--vcf` | VCF 基因型文件路径 | 互斥组 |
| `-bfile/--bfile` | PLINK 格式前缀 | 互斥组 |
| `-m/--method` | GRM 计算方法: `1`(居中), `2`(标准化) | 1 |
| `-npz/--npz` | 保存为压缩 NPZ 格式 (节省空间) | False |
| `-o/--out` | 输出目录 | 当前目录 |
| `-prefix/--prefix` | 输出文件前缀 | 自动生成 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.grm.id` | 样本 ID 列表 (与矩阵行/列对应) |
| `{prefix}.grm.txt` | GRM 矩阵 (文本格式) |
| `{prefix}.grm.npz` | GRM 矩阵 (压缩格式，如启用) |

**GRM 矩阵说明**:

- 对角线元素表示个体与自身的遗传关系，理论值为 1 (纯合) ~ 2 (完全杂合)
- 非对角线元素表示两个体间的遗传关系，范围 0~2

---

## PCA 主成分分析

### 模块说明

主成分分析 (Principal Component Analysis, PCA) 用于揭示群体遗传结构，常用于:

- 检测群体分层
- 识别离群个体
- 作为 GWAS 的协变量校正群体结构

### 使用方法

```bash
# 从 VCF 计算 PCA
jx pca --vcf data.vcf.gz -o out/

# 从 PLINK 计算
jx pca --bfile data -o out/

# 从预计算 GRM 计算
jx pca --grm prefix -o out/

# 从已有 PCA 结果可视化
jx pca --pcfile prefix --plot --plot3D -o out/

# 指定输出维度并可视化
jx pca --vcf data.vcf.gz --dim 5 --plot --plot3D -o out/

# 添加分组信息
jx pca --vcf data.vcf.gz --plot --group groups.txt -o out/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-vcf/--vcf` | VCF 基因型文件路径 | 互斥组 |
| `-bfile/--bfile` | PLINK 格式前缀 | 互斥组 |
| `-grm/--grm` | GRM 矩阵前缀 | 互斥组 |
| `-pcfile/--pcfile` | 已有 PCA 结果前缀 | 互斥组 |
| `-dim/--dim` | 输出主成分数量 | 3 |
| `-plot/--plot` | 生成 2D 散点图 | False |
| `-plot3D/--plot3D` | 生成 3D 旋转 GIF | False |
| `-group/--group` | 分组文件路径 (ID, 分组, [标签]) | None |
| `-color/--color` | 颜色方案索引 (0-6) | 1 |
| `-o/--out` | 输出目录 | 当前目录 |
| `-prefix/--prefix` | 输出文件前缀 | 自动生成 |

### 分组文件格式

```text
ID      Group   Label (可选)
indv1   PopA    Sample1
indv2   PopA    Sample2
indv3   PopB    Sample3
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.eigenvec` | PC 坐标矩阵 (样本数 × 维度数) |
| `{prefix}.eigenvec.id` | 样本 ID 列表 |
| `{prefix}.eigenval` | 各主成分的特征值 |
| `{prefix}.eigenvec.2D.pdf` | 2D 散点图 (如启用) |
| `{prefix}.eigenvec.3D.gif` | 3D 旋转 GIF (如启用) |
| `{prefix}.eigenvec.3D.gif` | 3D 旋转 GIF (如启用) |

---

## postGWAS 结果可视化

### 模块说明

对 GWAS 分析结果进行可视化，包括:

- **Manhattan 图**: 展示各染色体上 SNP 的 P 值分布
- **QQ 图**: 检验 GWAS 结果是否符合预期 (判断混杂)
- **SNP 注释**: 将显著位点关联到基因功能区域

### 使用方法

```bash
# 基本用法
jx postGWAS -f result.assoc.tsv

# 指定列名
jx postGWAS -f result.tsv -chr "chr" -pos "pos" -pvalue "P_wald"

# 设置显著性阈值
jx postGWAS -f result.tsv --threshold 1e-6

# 高亮特定 SNP 区域
jx postGWAS -f result.tsv --highlight snps.bed

# 注释显著 SNP
jx postGWAS -f result.tsv -a annotation.gff

# 扩展注释窗口 (±100kb)
jx postGWAS -f result.tsv -a annotation.gff --annobroaden 100

# 指定输出格式和颜色
jx postGWAS -f result.tsv --format pdf --color 2

# 批量处理多个文件
jx postGWAS -f results/*.tsv --threshold 1e-6 --out plots/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f/--file` | GWAS 结果文件路径 (支持 glob 模式) | 必填 |
| `-chr/--chr` | 染色体列名 | "#CHROM" |
| `-pos/--pos` | 位置列名 | "POS" |
| `-pvalue/--pvalue` | P 值列名 | "p" |
| `-threshold/--threshold` | 显著性阈值 | 0.05 / SNP 数 |
| `-noplot/--noplot` | 禁用绘图，仅输出注释 | False |
| `-color/--color` | 颜色方案 (0-6) | 0 |
| `-hl/--highlight` | BED 格式高亮区域文件 | None |
| `-format/--format` | 图像格式: pdf/png/svg/tif | "png" |
| `-a/--anno` | 注释文件路径 (GFF/GTF/BED) | None |
| `-ab/--annobroaden` | 注释扩展窗口 (kb) | None |
| `-descItem/--descItem` | GFF 描述字段的键名 | "description" |
| `-o/--out` | 输出目录 | 当前目录 |
| `-prefix/--prefix` | 日志文件前缀 | "JanusX" |
| `-t/--thread` | 并行线程数 (`-1` 表示全部可用核心) | -1 |

### 高亮文件格式

```text
chr1    start   end     name    description
chr1    1000000 1000000 GeneA   Description of GeneA
chr1    2000000 2000000 GeneB   Description of GeneB
```

### 注释文件格式

支持 GFF/GTF 或 BED 格式。GFF 示例:

```text
chr1    .       gene    1000000 2000000 .       +       .       ID=gene1;description=Example gene
chr1    .       mRNA    1000000 2000000 .       +       .       ID=mRNA1;Parent=gene1
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.manh.{format}` | Manhattan 图 |
| `{prefix}.qq.{format}` | QQ 图 |
| `{prefix}.{threshold}.anno.tsv` | 注释后的显著 SNP 列表 (如启用) |

---

## SIM 数据模拟

### 模块说明

生成模拟的基因型和表型数据，用于测试流程验证。

### 使用方法

```bash
# 基本用法：模拟 100k SNP，500 个体
jx sim 100 500 output_prefix

# 自定义 SNP 数量 (k 为单位)
jx sim 500 1000 mydata   # 500k SNPs, 1000 individuals
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `nsnp(k)` | SNP 数量 (以千为单位) |
| `nind` | 个体数量 |
| `outprefix` | 输出文件前缀 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `{prefix}.vcf.gz` | 模拟的基因型数据 (VCF 格式) |
| `{prefix}.pheno` | 完整表型文件 |
| `{prefix}.pheno.txt` | 简化表型格式 (与 GS 模块兼容) |

---

## 常见问题

### Q: 如何选择 GWAS 模型？

根据数据特征选择合适的模型:

| 数据特征 | 推荐模型 | 说明 |
|----------|----------|------|
| 大样本、无明显群体结构 | GLM | 速度快，内存占用低 |
| 有群体结构或亲缘关系 | LMM | 控制假阳性，推荐首选 |
| 需要平衡检验力和假阳性 | FarmCPU | 双重检验，控制混杂 |

### Q: 内存不足怎么办？

1. 使用 GLM 或 LMM 模型 (支持流式处理)
2. 减小 `--chunksize` 参数值
3. FarmCPU 需要全量加载基因型，内存要求较高

### Q: 如何复用计算结果？

JanusX 会自动缓存中间结果，避免重复计算:

| 缓存文件 | 对应模块 | 说明 |
|----------|----------|------|
| `{prefix}.k.{method}.npy` | GRM | 遗传关系矩阵 |
| `{prefix}.q.{dim}.txt` | PCA | 主成分矩阵 |
| `{prefix}.grm.bin` | GRM | 二进制格式矩阵 |

### Q: 输出结果为空？

检查以下几点:

1. 基因型和表型的样本 ID 是否匹配
2. 检查表型列是否存在缺失值
3. 使用 `--thread 1` 顺序执行模式进行调试
4. 查看日志文件排查具体错误

### Q: 如何使用多个协变量？

协变量文件格式与表型文件类似，但去除列名和行名称，行需要与基因型样本一一对应:

```text
0.1     -0.2    25
0.3     0.1     30
```

使用 `-c/--cov` 参数指定协变量文件。

### Q: 如何解读 QQ 图？

QQ 图用于检验 GWAS 结果是否存在混杂:

- 点沿对角线分布: 结果正常 (符合零假设)
- 右上角偏离对角线: 存在显著关联信号
- 整体偏离对角线: 可能存在群体分层或技术混杂

---

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
