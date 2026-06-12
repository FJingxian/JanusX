#include "kmc_wrapper.h"

#include <cstdint>
#include <cctype>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmc_file.h"
#include "kmc_runner.h"

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

KMC::InputFileType parse_input_type(const std::string& input_type_raw) {
    std::string s;
    s.reserve(input_type_raw.size());
    for (char c : input_type_raw) {
        s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (s == "fastq") {
        return KMC::InputFileType::FASTQ;
    }
    if (s == "fasta") {
        return KMC::InputFileType::FASTA;
    }
    if (s == "multiline-fasta" || s == "multiline_fasta" || s == "mlfasta") {
        return KMC::InputFileType::MULTILINE_FASTA;
    }
    throw std::runtime_error("Unsupported input_type: " + input_type_raw);
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
) {
    clear_error();
    if (input_files == nullptr || input_files_len == 0U) {
        set_error("input_files cannot be empty");
        return 0;
    }
    if (output_prefix == nullptr || output_prefix[0] == '\0') {
        set_error("output_prefix cannot be empty");
        return 0;
    }
    if (tmp_dir == nullptr || tmp_dir[0] == '\0') {
        set_error("tmp_dir cannot be empty");
        return 0;
    }
    if (input_type == nullptr || input_type[0] == '\0') {
        set_error("input_type cannot be empty");
        return 0;
    }
    if (out_stats == nullptr) {
        set_error("out_stats cannot be null");
        return 0;
    }
    if (kmer_len == 0U) {
        set_error("kmer_len must be > 0");
        return 0;
    }
    if (max_ram_gb == 0U) {
        set_error("max_ram_gb must be > 0");
        return 0;
    }

    try {
        std::vector<std::string> files;
        files.reserve(input_files_len);
        for (size_t idx = 0; idx < input_files_len; ++idx) {
            const char* raw = input_files[idx];
            if (raw == nullptr || raw[0] == '\0') {
                throw std::runtime_error("input_files contains an empty path");
            }
            files.emplace_back(raw);
        }

        KMC::Stage1Params stage1;
        KMC::NullLogger quiet_logger;
        KMC::NullPercentProgressObserver quiet_percent;
        KMC::NullProgressObserver quiet_progress;
        stage1.SetInputFiles(files)
            .SetTmpPath(std::string(tmp_dir))
            .SetKmerLen(kmer_len)
            .SetMaxRamGB(max_ram_gb)
            .SetInputFileType(parse_input_type(std::string(input_type)))
            .SetCanonicalKmers(canonical != 0)
            .SetVerboseLogger(&quiet_logger)
            .SetWarningsLogger(&quiet_logger)
            .SetPercentProgressObserver(&quiet_percent)
            .SetProgressObserver(&quiet_progress);
        if (threads > 0U) {
            stage1.SetNThreads(threads);
        }

        KMC::Stage2Params stage2;
        stage2.SetMaxRamGB(max_ram_gb)
            .SetOutputFileName(std::string(output_prefix))
            .SetOutputFileType(KMC::OutputFileType::KMC)
            .SetCutoffMin(cutoff_min)
            .SetCutoffMax(cutoff_max)
            .SetCounterMax(counter_max);
        if (threads > 0U) {
            stage2.SetNThreads(threads);
        }

        KMC::Runner runner;
        auto stage1_results = runner.RunStage1(stage1);
        auto stage2_results = runner.RunStage2(stage2);

        out_stats->stage1_time_s = static_cast<double>(stage1_results.time);
        out_stats->stage2_time_s = static_cast<double>(stage2_results.time);
        out_stats->n_sequences = static_cast<uint64_t>(stage1_results.nSeqences);
        out_stats->tmp_size_stage1 = static_cast<uint64_t>(stage1_results.tmpSize);
        out_stats->n_total_kmers = static_cast<uint64_t>(stage2_results.nTotalKmers);
        out_stats->n_unique_kmers = static_cast<uint64_t>(stage2_results.nUniqueKmers);
        out_stats->n_below_cutoff_min = static_cast<uint64_t>(stage2_results.nBelowCutoffMin);
        out_stats->n_above_cutoff_max = static_cast<uint64_t>(stage2_results.nAboveCutoffMax);
        out_stats->max_disk_usage = static_cast<uint64_t>(stage2_results.maxDiskUsage);
        return 1;
    } catch (const std::exception& err) {
        set_error(err.what());
        return 0;
    } catch (...) {
        set_error("unknown C++ exception in jx_kmc_count_run");
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
