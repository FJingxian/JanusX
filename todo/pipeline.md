# fastq2count pipeline (RNA-seq)

## Goal
Build a reproducible `FASTQ -> gene count/FPKM/TPM` workflow in launcher, with step-wise resume behavior consistent with `fastq2vcf` style.

## CLI (launcher)

```bash
jx fastq2count \
  -r /path/to/reference.fa \
  -a /path/to/annotation.gtf \
  -i /path/to/fastq_dir \
  -w /path/to/workdir \
  [-from-step 1] [-to-step 4] \
  [-strandness FR] [-t 16] \
  [-feature-type exon] [-gene-attr gene_id]
```

Notes:
- `-from-step/-to-step` are inclusive.
- `-i/--in` is always required and must point to the input directory for the selected `-from-step`.
- Backend follows launcher behavior: use `csub` if available, otherwise `nohup`.

## Directory layout

```
workdir/
  1.clean/
    {sample}.R1.clean.fastq.gz
    {sample}.R2.clean.fastq.gz
    qc/{sample}.html
    qc/{sample}.json
  2.index/
    reference.index.ok
    reference.ss
    reference.exon
    reference.*.ht2 or reference.*.ht2l
  3.mapping/
    {sample}.bam
    {sample}.bam.bai
    {sample}.hisat2.log
  4.count/
    gene_counts.txt
    gene_counts.fpkm.tsv
    gene_counts.tpm.tsv
  log/
  tmp/
```

## Step design

### Step 1: fastp QC
Input:
- Paired FASTQ (`R1/R2`) from `-i`.

Output:
- `1.clean/{sample}.R1.clean.fastq.gz`
- `1.clean/{sample}.R2.clean.fastq.gz`
- `1.clean/qc/{sample}.html`
- `1.clean/qc/{sample}.json`

### Step 2: HISAT2 index build
Input:
- `reference.fa`
- annotation (`.gtf/.gff/.gff3`)

Logic:
- If `hisat2_extract_splice_sites.py` and `hisat2_extract_exons.py` are available, generate:
  - `2.index/reference.ss`
  - `2.index/reference.exon`
  and build with `--ss/--exon`.
- If helper scripts are absent, fallback to plain `hisat2-build` and keep warning in stderr.

Output:
- HISAT2 index files under `2.index/`
- success marker: `2.index/reference.index.ok`

### Step 3: HISAT2 alignment + BAM sort/index
Input:
- Cleaned FASTQ from step 1
- HISAT2 index from step 2

Command style:
- `hisat2 --new-summary --rna-strandness ... | samtools sort`
- `samtools index`

Output:
- `3.mapping/{sample}.bam`
- `3.mapping/{sample}.bam.bai`
- `3.mapping/{sample}.hisat2.log`

### Step 4: featureCounts + FPKM/TPM
Input:
- all sample BAM from step 3
- annotation (`-a`)

Output:
- raw count: `4.count/gene_counts.txt`
- normalized tables:
  - `4.count/gene_counts.fpkm.tsv`
  - `4.count/gene_counts.tpm.tsv`

Normalization formula:
- `FPKM = count * 1e9 / (gene_length_bp * library_size_reads)`
- `TPM = (count / gene_length_kb) / sum(count / gene_length_kb) * 1e6`

## Input pairing rules
Recognized FASTQ suffixes:
- `.fastq.gz`, `.fq.gz`, `.fastq`, `.fq`

Recognized read tokens:
- `_R1/_R2`, `.R1/.R2`, `_READ1/_READ2`, `_1/_2`, `.1/.2`, `_R1_001/_R2_001`

If no valid pairs are detected, pipeline exits with an explicit suffix/token hint.

## Failure and resume behavior
- Missing required tools are reported before execution (green/yellow line status).
- Existing outputs are reused (`skip_if_outputs_exist=true`).
- Run subset with `-from-step/-to-step` for partial rerun.

## Required tools
- `fastp` (supports `jx fastp` wrapper or system `fastp`)
- `hisat2-build`
- `hisat2`
- `samtools`
- `featureCounts`
- `python3` (used for FPKM/TPM table generation)

## popanalysis

```bash

```
