# FvLMM macOS 扫描性能优化设计

## 目标

在不改变 FvLMM 统计结果、输出顺序和 CLI 行为的前提下，确认 macOS Accelerate 后端的实际性能上限，并依次优化 JanusX GWAS 的 FvLMM 扫描路径。

## 已确认的现状

- PLINK BED 的单模型 FvLMM 走 `src/stats/fvlmm.rs::fvlmm_assoc_bed_to_tsv_f32`。
- 扫描主循环是 `decode/count -> SNP block × U^T -> fixed-lambda association -> TSV`。
- 旋转阶段默认使用一个 CBLAS SGEMM；macOS Accelerate 的 `BLASSetThreading` 只接受单线程/多线程模式，不接受精确线程数。
- 现有 `JX_FVLMM_CHUNK_PROFILING` 不覆盖 BED unified path，`JX_FVLMM_PACKED_STAGE_TIMING` 只覆盖 packed path。
- Python 的 FvLMM stage policy 与 Rust 通过 `JX_MLM_RUST_THREADS` 读取的 stage 线程存在耦合；`blas_t_rayon_1` 可能把 Rust 投影线程一并降为 1。

## 设计范围

### 阶段 1：BED unified path 可观测性

增加可选环境变量 `JX_FVLMM_BED_STAGE_TIMING=1`，在一次 Rust BED 扫描结束时输出：

- decode/count、projection、association、TSV、总耗时及未归因耗时；
- stage active time / wall time；
- rows、samples、covariate columns、block rows；
- backend、requested/projection/association Rayon 线程和 pipeline 状态。

计时默认关闭，不改变输出文件；已有 packed/chunk profiling 保持兼容。

### 阶段 2：线程语义解耦

Rust FvLMM 扫描显式接收 projection 和 association 线程预算。Python stage context 不再通过单一 `JX_MLM_RUST_THREADS` 隐式改变两个 stage；兼容旧环境变量，但显式的 `JX_FVLMM_PROJ_THREADS` 和 `JX_FVLMM_ASSOC_THREADS` 优先级最高。

`full` 默认使用 CLI 线程数；`generic`/`blas_t_rayon_1` 的日志和实际 Rust 行为必须一致。对于 Accelerate，projection 线程数只表示是否启用 BLAS 多线程，实际 worker 数由 Accelerate 决定。

### 阶段 3：可选 tiled-CBLAS projection

保留当前单个大 SGEMM 作为默认内核，新增显式 opt-in 的 tiled-CBLAS 内核：

- 将 SNP block 按行拆成足够多的 tile；
- tile 内调用单线程 CBLAS，tile 间使用 projection Rayon pool；
- 根据 `rows × n` 和线程预算避免产生过小的 GEMM；
- 结果布局、f32 乘法和输出顺序保持一致。

只有当同一数据上的墙钟时间稳定优于默认内核且数值误差通过阈值，才考虑改变默认选择。CPU 百分比不是单独的成功标准。

### 阶段 4：融合与多性状复用评估

先使用阶段计时决定是否实现：

- projection 与 fixed-lambda association 的块内融合，减少 `rot_block` 的写回和再次读取；
- 多个共享样本掩码/特征空间的性状复用 `SNP × U^T`。

若融合不能在基准上同时降低墙钟时间并保持数值一致，则只保留诊断，不引入复杂重构。

## 正确性约束

- BED 和 packed 结果列、SNP 顺序、缺失率、beta、SE、p 值保持兼容。
- 默认内核输出必须 byte-for-byte 保持不变；tiled 内核允许浮点舍入差异，但 beta/SE 相对误差不超过 `1e-5`，p 值绝对误差不超过 `1e-10`（有限值）。
- 计时和线程设置不能改变随机数、过滤或 TSV 写入顺序。
- 现有用户未提交改动不得被重写或纳入本次优化。

## 验证矩阵

1. Rust 单元测试：线程策略、tile 划分边界、空块/小块。
2. PyO3 kernel 回归：默认与 opt-in tiled 的 beta/SE/p 输出比较。
3. CLI 示例：`example/~mouse_hs1940.snp0`，`-n 0 -fvlmm -t 1,2,4,8`，记录 wall/CPU/RSS。
4. 后端矩阵：至少当前 macOS Accelerate；若可用，补充 OpenBLAS 线程语义测试。
5. 工程检查：`cargo fmt --all -- --check`、聚焦 `cargo test`、编辑 Python 的 `py_compile`、重建 release PyO3 扩展后再运行 `jx`。

## 回滚策略

- 阶段计时通过环境变量关闭即可。
- tiled 内核必须显式 opt-in，默认路径不变。
- 线程语义修复保留旧变量读取，异常配置回退到 CLI 线程数。
