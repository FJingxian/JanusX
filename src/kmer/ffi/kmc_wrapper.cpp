#include "kmc_wrapper.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmc_file.h"

namespace {

thread_local std::string JX_KMC_LAST_ERROR;

struct JxKmcReader {
    CKMCFile db;
    CKMCFileInfo info{};
    CKmerAPI kmer;
    bool eof = false;

    explicit JxKmcReader(uint32_t k_hint)
        : kmer(k_hint > 0 ? k_hint : 1) {}
};

void clear_error() {
    JX_KMC_LAST_ERROR.clear();
}

void set_error(const std::string& msg) {
    JX_KMC_LAST_ERROR = msg;
}

uint64_t encode_kmer_direct_u64(const CKmerAPI& kmer, std::vector<uint64_t>& words_buf) {
    kmer.to_long(words_buf);
    if (words_buf.size() != 1U) {
        throw std::runtime_error("current wrapper only supports k <= 31");
    }
    return words_buf[0];
}

}  // namespace

extern "C" {

JxKmcHandle jx_kmc_open(const char* db_prefix, uint32_t k_hint) {
    clear_error();
    if (db_prefix == nullptr || db_prefix[0] == '\0') {
        set_error("KMC prefix cannot be empty");
        return nullptr;
    }

    try {
        std::unique_ptr<JxKmcReader> reader(new JxKmcReader(k_hint));
        if (!reader->db.OpenForListing(std::string(db_prefix))) {
            set_error(std::string("failed to open KMC database: ") + db_prefix);
            return nullptr;
        }
        if (!reader->db.Info(reader->info)) {
            reader->db.Close();
            set_error(std::string("failed to read KMC database info: ") + db_prefix);
            return nullptr;
        }
        if (reader->info.kmer_length == 0U || reader->info.kmer_length > 31U) {
            reader->db.Close();
            set_error("current wrapper only supports 1 <= k <= 31");
            return nullptr;
        }
        if (k_hint > 0 && reader->info.kmer_length != k_hint) {
            reader->db.Close();
            set_error("KMC k-mer length mismatch with requested k_hint");
            return nullptr;
        }
        reader->kmer = CKmerAPI(reader->info.kmer_length);
        reader->eof = false;
        return static_cast<JxKmcHandle>(reader.release());
    } catch (const std::exception& err) {
        set_error(err.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception in jx_kmc_open");
        return nullptr;
    }
}

int jx_kmc_info(JxKmcHandle handle, JxKmcDbInfo* out_info) {
    clear_error();
    if (handle == nullptr || out_info == nullptr) {
        set_error("invalid null pointer in jx_kmc_info");
        return 0;
    }
    try {
        JxKmcReader* reader = static_cast<JxKmcReader*>(handle);
        out_info->kmer_length = reader->info.kmer_length;
        out_info->min_count = reader->info.min_count;
        out_info->max_count = reader->info.max_count;
        out_info->total_kmers = reader->info.total_kmers;
        out_info->counter_size = reader->info.counter_size;
        out_info->both_strands = reader->info.both_strands ? 1u : 0u;
        for (unsigned char& slot : out_info->reserved) {
            slot = 0u;
        }
        return 1;
    } catch (const std::exception& err) {
        set_error(err.what());
        return 0;
    } catch (...) {
        set_error("unknown C++ exception in jx_kmc_info");
        return 0;
    }
}

size_t jx_kmc_read_batch_u64(
    JxKmcHandle handle,
    uint64_t* kmer_buf,
    uint32_t* count_buf,
    size_t max_records
) {
    clear_error();
    if (handle == nullptr || kmer_buf == nullptr || count_buf == nullptr || max_records == 0U) {
        return 0U;
    }

    try {
        JxKmcReader* reader = static_cast<JxKmcReader*>(handle);
        size_t written = 0U;
        uint64_t count = 0U;
        std::vector<uint64_t> words_buf;
        words_buf.reserve(1U);
        while (written < max_records && reader->db.ReadNextKmer(reader->kmer, count)) {
            kmer_buf[written] = encode_kmer_direct_u64(reader->kmer, words_buf);
            count_buf[written] = static_cast<uint32_t>(count);
            ++written;
        }
        reader->eof = (written < max_records);
        return written;
    } catch (const std::exception& err) {
        set_error(err.what());
        return 0U;
    } catch (...) {
        set_error("unknown C++ exception in jx_kmc_read_batch_u64");
        return 0U;
    }
}

int jx_kmc_is_eof(JxKmcHandle handle) {
    if (handle == nullptr) {
        return 1;
    }
    try {
        JxKmcReader* reader = static_cast<JxKmcReader*>(handle);
        return reader->eof ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void jx_kmc_close(JxKmcHandle handle) {
    if (handle == nullptr) {
        return;
    }
    try {
        JxKmcReader* reader = static_cast<JxKmcReader*>(handle);
        reader->db.Close();
        delete reader;
    } catch (...) {
    }
}

const char* jx_kmc_last_error(void) {
    return JX_KMC_LAST_ERROR.c_str();
}

}  // extern "C"
