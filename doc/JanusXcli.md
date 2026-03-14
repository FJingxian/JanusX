# JanusX CLI 文档

版本基线：`v1.0.13`  
入口命令：`jx`

## 1. 总览

**JanusX 的命令行由两层组成**：

1. `launcher`（Rust）：负责环境、更新、DLC 工具链 (for BSA, RNAseq流程) 等。
2. `python core`：负责 `gwas/gs/postgwas/postbsa/...` 等统计、建模与绘图模块。

**基础用法**：

```bash
jx -h
jx -v
jx <module> -h
```

## 2. 全局 Flags

```text
-h           显示 launcher 帮助
-v           显示版本、构建时间
-update      更新 JanusX 依赖（dlc/latest/local）
-upgrade     升级 launcher（并触发核心依赖更新）
-list        列出模块或 DLC 工具（module/dlc）
-clean       清理历史数据库，或按 history_id 删除单条记录
-uninstall   卸载 JanusX (依赖及其所有环境)
```

**常用指令**：

```bash
jx -update latest # 更新 Github 上最新版python核心
jx -update dlc # 检测与安装 pipeline 所需的工具链
jx -list dlc # 展示已安装工具链
jx -upgrade # 更新 JanusX 启动器
```

## 3. 模块列表

### 3.1 全基因组关联分析

```text
grm        构建 GRM
pca        群体结构 PCA
gwas       GWAS（LM/LMM/FarmCPU/fastLMM）
postgwas   GWAS 可视化与注释
```

### 3.2 全基因组选择

```text
gs      基因组选择建模
reml    REML-BLUP 方差分量与遗传力
```

### 3.3 GARFIELD

```text
garfield      RF 关联分析
postgarfield  GARFIELD 后处理
```

### 3.4 BSA

```text
postbsa   BSA 结果处理与绘图
```

### 3.5 流水线与工具

```text
fastq2count   FASTQ 到 gene count/FPKM/TPM
fastq2vcf     FASTQ 到 VCF
hybrid        亲本组合杂交基因型构建
gformat       基因型格式转换
gmerge        多基因型数据合并个体
webui         启动 WebUI (测试版)
```

### 3.6 Benchmark

```text
sim          快速模拟 (生成随机基因型+表型)
simulation   扩展模拟流程 (基于基因型模拟表型)
```

## 4. 输入文件格式（重点）

### 4.1 基因型输入（`-vcf/-hmp/-bfile/-file`）

```text
-vcf     *.vcf / *.vcf.gz，标准 VCF，样本在 FORMAT 之后的列
-hmp     *.hmp / *.hmp.gz，标准 HapMap 文本
-bfile   PLINK 前缀，标准bed格式，必须同时存在 prefix.bed/.bim/.fam
-file    前缀，JanusX 内建格式，必须同时存在 prefix.npy/.id 或 prefix.txt/.id，可选prefix.site 提供位点信息
```

**标准VCF**：
`*.vcf` `*.vcf.gz`

```vcf
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	JX_A01	JX_A02	JX_A03
1	200153	rsJX000001	G	A	.	PASS	.	GT	0/1	0/0	1/1
1	200987	rsJX000002	T	C	.	PASS	.	GT	0/0	0/1	0/0
1	205432	rsJX000003	C	G	.	PASS	.	GT	1/1	0/1	0/0
```

**标准hmp格式**：
`*.vcf` `*.vcf.gz`

```hmp
rs#	alleles	chrom	pos	strand	assembly#	center	protLSID	assayLSID	panelLSID	QCcode	Ind1	Ind2	Ind3
SNP_1	A/G	1	1001	+	NA	NA	NA	NA	NA	NA	AA	AG	GG
SNP_2	C/T	1	2050	+	NA	NA	NA	NA	NA	NA	CT	CC	TT
SNP_3	G/T	2	3300	+	NA	NA	NA	NA	NA	NA	GG	NN	GT
```

**标准bed格式**:
`prefix.bed/.bim/.fam` [Details](http://zzz.bwh.harvard.edu/plink/data.shtml#bed)

**JanusX 内建格式**：
`prefix.txt` (required)

```text
1	0	1
1	2	1
1	0	0
```

`prefix.id` (required)

```text
Sample_001
Sample_002
Sample_003
```

`prefix.site` (optional)

```text
1	10583	A	G
1	10611	C	T
1	13302	G	A
```

### 4.2 表型文件（`-p/--pheno`）

```text
SampleID	trait1	trait2
S1	1.23	5.67
S2	2.34	6.78
S3	3.45	7.89
```

`gwas/gs` 均兼容 `tab/comma/空白` 分隔，推荐 TSV 以便复现与人工检查。

可通过 `-n` 指定特定表型

`-n 3`: 单列
`-n 1,3,5`: 列表
`-n 1:10`、`-n 1-10`: 范围
`-n 1 -n 3`: 重复传参

### 4.4 GRM 输入（`gwas -k/--grm`）

`-k 1`：内置 centered GRM（由基因型构建）。
`-k 2`：内置 standardized GRM（由基因型构建）。
`-k <grm_file>`：外部 GRM 文件。

**GRM兼容格式**:
`grm.txt` 或 `grm.npy`

```text
2.2	0.0	1.1
0.0	2.1	1.5
1.1	1.5	2.1
```

`grm.txt.id` 或 `grm.npy.id` (optional)

```text
Sample_001
Sample_002
Sample_003
```

### 4.5 Q/PC 输入（`gwas -q/--qcov`）

`-q 0`：不使用 PCA 协变量。
`-q <正整数>`：从 GRM 计算前 N 个 PC。
`-q <q_file>` 兼容性已下线；外部协变量矩阵请使用 `-c <file>`。

### 4.6 协变量输入（`gwas -c/--cov`）

`-c` 可重复，支持两种输入：

1. 协变量文件路径。
2. SNP 位点 token：`chr:pos` 或 `chr:start:end`（其中 `start` 必须等于 `end`）。

协变量文件格式：

1. 格式 A：第一列样本 ID，后续列为数值协变量。
2. 格式 B：纯数值矩阵（无 ID），行顺序必须与基因型样本顺序一致。

注意：

1. 协变量列必须是数值；分类变量请先自行独热编码。
2. 多个协变量源会按样本交集对齐后拼接。

### 4.7 样本 ID 对齐与常见坑

1. 所有模块最终都按样本 ID 取交集；交集过小会显著降低有效样本量。
2. `-file` 模式下最关键是 `prefix.id` 与矩阵列严格同序。
3. 外部 GRM/Q/cov 若不带 ID，将退化为“按行顺序对齐”，更容易出错。
4. 若过滤后位点几乎全被去除，可能出现 `Invalid GRM denominator (varsum<=0)`。

## 5. 核心命令速查

### 5.1 GWAS / GS / REML

```bash
# GWAS
jx gwas -vcf geno.vcf.gz -p pheno.tsv -lmm -o outdir

# GS
jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5 -o outdir

# REML
jx reml -file test.reml.txt -rh 1,2,3 -n 4 -o outdir -prefix reml
```

### 5.2 后处理

```bash
# postgwas
jx postgwas -gwasfile result.tsv -manh -qq -o outdir

# postbsa（支持通配）
jx postbsa -file '4.merge/Merge.*.SNP.tsv' -b1 Bulk1 -b2 Bulk2 -t 8
```

### 5.3 基因型处理

```bash
# 格式转换
jx gformat -vcf geno.vcf.gz -fmt npy -o out -prefix panel

# 合并
jx gmerge -vcf a.vcf.gz b.vcf.gz -fmt vcf -o out -prefix merge

# 杂交矩阵
jx hybrid -file geno_prefix -p1 p1.list -p2 p2.list -fmt npy -o out -prefix hybrid
```

### 5.4 流水线

### `fastq2vcf`

```bash
jx fastq2vcf -r ref.fa -i raw_fastq_dir -w workdir
jx fastq2vcf -r ref.fa -i 3.gvcf -w workdir -from-step 4 -to-step 6
```

步骤（当前实现）：

1. `fastp`
2. `bwamem+markdup`
3. `bam2gvcf`
4. `gvcf2gtprep`（合并 gVCF + genotype + GT 准备）
5. `impute_filter`
6. `mergevcf`

### `fastq2count`

```bash
jx fastq2count -r ref.fa -a anno.gtf -i fastq_dir -w workdir
jx fastq2count -r ref.fa -a anno.gff3 -i work/3.mapping -w work -from-step 3 -to-step 4
```

内部步骤：

1. `fastp`
2. `hisat2-index`
3. `hisat2-align`
4. `featurecounts`

显示上常以三段逻辑展示：`fastp+hisat2-index`、`hisat2-align`、`featurecounts`。

## 6. DLC 工具链

`fastq2vcf/fastq2count` 依赖外置工具（docker/conda/env/singularity 路由）。

```bash
jx -update dlc
jx -list dlc
```

当提示 `Missing DLC tool wrappers` 时，优先执行：

```bash
jx -update dlc
```

## 7. 更新与升级策略

1. `jx -update latest`：源码更新 Python core（当前策略默认重装 latest）。
2. `jx -upgrade`：源码升级 launcher，再执行 core 更新逻辑。
