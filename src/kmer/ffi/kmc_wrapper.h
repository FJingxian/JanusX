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

JxKmcHandle jx_kmc_open(const char* db_prefix, uint32_t k_hint);
int jx_kmc_info(JxKmcHandle handle, JxKmcDbInfo* out_info);
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
