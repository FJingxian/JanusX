#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* JxKmcHandle;

typedef struct JxKmcDbInfo {
    uint32_t kmer_length;
    uint32_t min_count;
    uint64_t max_count;
    uint64_t total_kmers;
    uint32_t counter_size;
    uint8_t both_strands;
    uint8_t reserved[7];
} JxKmcDbInfo;

typedef struct JxKmcCountStats {
    double stage1_time_s;
    double stage2_time_s;
    uint64_t n_sequences;
    uint64_t tmp_size_stage1;
    uint64_t n_total_kmers;
    uint64_t n_unique_kmers;
    uint64_t n_below_cutoff_min;
    uint64_t n_above_cutoff_max;
    uint64_t max_disk_usage;
} JxKmcCountStats;

JxKmcHandle jx_kmc_open(const char* db_prefix, uint32_t k_hint);
int jx_kmc_info(JxKmcHandle handle, JxKmcDbInfo* out_info);
int jx_kmc_count_run(
    const char* const* input_files,
    size_t input_files_len,
    const char* output_prefix,
    const char* tmp_dir,
    uint32_t kmer_len,
    uint32_t threads,
    uint32_t max_ram_gb,
    uint64_t cutoff_min,
    uint64_t cutoff_max,
    uint64_t counter_max,
    int canonical,
    const char* input_type,
    JxKmcCountStats* out_stats
);
size_t jx_kmc_read_batch_u64(
    JxKmcHandle handle,
    uint64_t* kmer_buf,
    uint32_t* count_buf,
    size_t max_records
);
int jx_kmc_is_eof(JxKmcHandle handle);
void jx_kmc_close(JxKmcHandle handle);
const char* jx_kmc_last_error(void);

#ifdef __cplusplus
}
#endif
