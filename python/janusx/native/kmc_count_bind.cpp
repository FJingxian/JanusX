#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "kmc_file.h"
#include "kmc_runner.h"

namespace py = pybind11;
static constexpr char BIN01_MAGIC[8] = {'J', 'X', 'B', 'I', 'N', '0', '0', '1'};
static constexpr char BIN_SITE_MAGIC[8] = {'J', 'X', 'B', 'S', 'I', 'T', 'E', '1'};
static constexpr std::uint64_t BIN_RESERVED = 0ULL;

static KMC::InputFileType parse_input_type(const std::string& input_type_raw) {
    std::string s;
    s.reserve(input_type_raw.size());
    for (char c : input_type_raw) {
        if (c >= 'A' && c <= 'Z') {
            s.push_back(static_cast<char>(c - 'A' + 'a'));
        } else {
            s.push_back(c);
        }
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

static void write_u64_le(std::ofstream& out, std::uint64_t v) {
    const char b[8] = {
        static_cast<char>(v & 0xFF),
        static_cast<char>((v >> 8) & 0xFF),
        static_cast<char>((v >> 16) & 0xFF),
        static_cast<char>((v >> 24) & 0xFF),
        static_cast<char>((v >> 32) & 0xFF),
        static_cast<char>((v >> 40) & 0xFF),
        static_cast<char>((v >> 48) & 0xFF),
        static_cast<char>((v >> 56) & 0xFF),
    };
    out.write(b, 8);
}

static void write_u16_le(std::ofstream& out, std::uint16_t v) {
    const char b[2] = {
        static_cast<char>(v & 0xFF),
        static_cast<char>((v >> 8) & 0xFF),
    };
    out.write(b, 2);
}

static std::uint8_t encode_base_2bit(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A':
            return 0;
        case 'T':
            return 1;
        case 'C':
            return 2;
        case 'G':
            return 3;
        default:
            throw std::runtime_error(std::string("Unsupported base in k-mer for bin.site: '") + c + "'");
    }
}

static std::vector<std::uint8_t> encode_kmer_site_record_2bit_bytes(const std::string& kmer) {
    const std::size_t klen = kmer.size();
    if (klen == 0) {
        throw std::runtime_error("Empty k-mer encountered while writing bin.site.");
    }
    if (klen > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
        throw std::runtime_error("k-mer length exceeds uint16 limit for bin.site.");
    }
    std::vector<std::uint8_t> rec;
    rec.resize(2 + ((klen + 3) / 4), 0u);
    const std::uint16_t u16_len = static_cast<std::uint16_t>(klen);
    rec[0] = static_cast<std::uint8_t>(u16_len & 0xFFu);
    rec[1] = static_cast<std::uint8_t>((u16_len >> 8) & 0xFFu);
    for (std::size_t i = 0; i < klen; ++i) {
        const std::uint8_t code = encode_base_2bit(kmer[i]);
        rec[2 + (i / 4)] |= static_cast<std::uint8_t>((code & 0x3u) << ((i % 4) * 2));
    }
    return rec;
}

static std::uint32_t normalize_threads(std::uint32_t threads) {
    std::uint32_t t = threads;
    if (t == 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        t = (hw > 0U) ? static_cast<std::uint32_t>(hw) : 1U;
    }
    if (t == 0U) {
        t = 1U;
    }
    return t;
}

static void encode_site_records_parallel(
    const std::vector<std::string>& kmers,
    std::vector<std::vector<std::uint8_t>>& out_records,
    std::uint32_t threads
) {
    const std::size_t n = kmers.size();
    out_records.clear();
    out_records.resize(n);
    if (n == 0) {
        return;
    }
    const std::uint32_t t_norm = normalize_threads(threads);
    if (t_norm <= 1U || n < 512ULL) {
        for (std::size_t i = 0; i < n; ++i) {
            out_records[i] = encode_kmer_site_record_2bit_bytes(kmers[i]);
        }
        return;
    }
    const std::size_t workers = (std::min)(static_cast<std::size_t>(t_norm), n);
    std::vector<std::thread> pool;
    pool.reserve(workers);
    std::exception_ptr eptr = nullptr;
    std::mutex eptr_mtx;
    auto worker_fn = [&](std::size_t begin, std::size_t end) {
        try {
            for (std::size_t i = begin; i < end; ++i) {
                out_records[i] = encode_kmer_site_record_2bit_bytes(kmers[i]);
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(eptr_mtx);
            if (!eptr) {
                eptr = std::current_exception();
            }
        }
    };
    const std::size_t base = n / workers;
    const std::size_t rem = n % workers;
    std::size_t cur = 0;
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t span = base + ((w < rem) ? 1ULL : 0ULL);
        const std::size_t begin = cur;
        const std::size_t end = begin + span;
        cur = end;
        pool.emplace_back(worker_fn, begin, end);
    }
    for (auto& th : pool) {
        if (th.joinable()) {
            th.join();
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

static void write_bin01_header(std::ofstream& out, std::uint64_t n_sites, std::uint64_t n_samples) {
    out.seekp(0, std::ios::beg);
    out.write(BIN01_MAGIC, 8);
    write_u64_le(out, n_sites);
    write_u64_le(out, n_samples);
    write_u64_le(out, BIN_RESERVED);
}

static void write_bin_site_header(std::ofstream& out, std::uint64_t n_sites) {
    out.seekp(0, std::ios::beg);
    out.write(BIN_SITE_MAGIC, 8);
    write_u64_le(out, n_sites);
    write_u64_le(out, BIN_RESERVED);
}

static void write_kmer_site_record_2bit(std::ofstream& out, const std::string& kmer) {
    std::vector<std::uint8_t> rec = encode_kmer_site_record_2bit_bytes(kmer);
    out.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(rec.size()));
}

static py::dict kmc_count(
    const std::vector<std::string>& input_files,
    const std::string& output_prefix,
    const std::string& tmp_dir = ".",
    std::uint32_t kmer_len = 31,
    std::uint32_t threads = 0,
    std::uint32_t max_ram_gb = 12,
    std::uint64_t cutoff_min = 2,
    std::uint64_t cutoff_max = 1000000000ULL,
    std::uint64_t counter_max = 255ULL,
    bool canonical = true,
    const std::string& input_type = "fastq"
) {
    if (input_files.empty()) {
        throw std::runtime_error("input_files cannot be empty.");
    }
    if (output_prefix.empty()) {
        throw std::runtime_error("output_prefix cannot be empty.");
    }
    if (kmer_len == 0) {
        throw std::runtime_error("kmer_len must be > 0.");
    }
    if (max_ram_gb == 0) {
        throw std::runtime_error("max_ram_gb must be > 0.");
    }

    KMC::Stage1Params stage1;
    KMC::NullLogger quiet_logger;
    KMC::NullPercentProgressObserver quiet_percent;
    KMC::NullProgressObserver quiet_progress;
    stage1.SetInputFiles(input_files)
        .SetTmpPath(tmp_dir)
        .SetKmerLen(kmer_len)
        .SetMaxRamGB(max_ram_gb)
        .SetInputFileType(parse_input_type(input_type))
        .SetCanonicalKmers(canonical)
        .SetVerboseLogger(&quiet_logger)
        .SetWarningsLogger(&quiet_logger)
        .SetPercentProgressObserver(&quiet_percent)
        .SetProgressObserver(&quiet_progress);
    if (threads > 0) {
        stage1.SetNThreads(threads);
    }

    KMC::Stage2Params stage2;
    stage2.SetMaxRamGB(max_ram_gb)
        .SetOutputFileName(output_prefix)
        .SetOutputFileType(KMC::OutputFileType::KMC)
        .SetCutoffMin(cutoff_min)
        .SetCutoffMax(cutoff_max)
        .SetCounterMax(counter_max);
    if (threads > 0) {
        stage2.SetNThreads(threads);
    }

    KMC::Runner runner;
    auto stage1_results = runner.RunStage1(stage1);
    auto stage2_results = runner.RunStage2(stage2);

    py::dict out;
    out["stage1_time_s"] = stage1_results.time;
    out["stage2_time_s"] = stage2_results.time;
    out["n_sequences"] = stage1_results.nSeqences;
    out["tmp_size_stage1"] = stage1_results.tmpSize;
    out["n_total_kmers"] = stage2_results.nTotalKmers;
    out["n_unique_kmers"] = stage2_results.nUniqueKmers;
    out["n_below_cutoff_min"] = stage2_results.nBelowCutoffMin;
    out["n_above_cutoff_max"] = stage2_results.nAboveCutoffMax;
    out["max_disk_usage"] = stage2_results.maxDiskUsage;
    return out;
}

static void write_npy_f32_header(std::ofstream& out, std::uint64_t rows, std::uint64_t cols) {
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const char version[2] = {1, 0};
    out.write(version, 2);

    std::ostringstream oss;
    oss << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols
        << "), }";
    std::string header = oss.str();

    const std::size_t preamble = 6 + 2 + 2;
    std::size_t header_len = header.size() + 1;
    const std::size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();

    if (header_len > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
        throw std::runtime_error("NPY header too large for v1.0 format.");
    }
    const std::uint16_t hlen = static_cast<std::uint16_t>(header_len);
    const char hlen_le[2] = {
        static_cast<char>(hlen & 0xFF),
        static_cast<char>((hlen >> 8) & 0xFF),
    };
    out.write(hlen_le, 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}

static py::dict kmc_db_info(const std::string& kmc_prefix) {
    if (kmc_prefix.empty()) {
        throw std::runtime_error("kmc_prefix cannot be empty.");
    }
    CKMCFile kmc;
    if (!kmc.OpenForListing(kmc_prefix)) {
        throw std::runtime_error("Failed to open KMC database for listing: " + kmc_prefix);
    }
    CKMCFileInfo info{};
    if (!kmc.Info(info)) {
        kmc.Close();
        throw std::runtime_error("Failed to read KMC database info: " + kmc_prefix);
    }
    kmc.Close();

    py::dict out;
    out["kmer_length"] = info.kmer_length;
    out["min_count"] = info.min_count;
    out["max_count"] = info.max_count;
    out["total_kmers"] = info.total_kmers;
    out["counter_size"] = info.counter_size;
    out["both_strands"] = info.both_strands;
    return out;
}

static py::dict kmc_export_janusx_single(
    const std::string& kmc_prefix,
    const std::string& out_prefix,
    const std::string& sample_id = "sample"
) {
    if (kmc_prefix.empty()) {
        throw std::runtime_error("kmc_prefix cannot be empty.");
    }
    if (out_prefix.empty()) {
        throw std::runtime_error("out_prefix cannot be empty.");
    }
    if (sample_id.empty()) {
        throw std::runtime_error("sample_id cannot be empty.");
    }

    CKMCFile kmc;
    if (!kmc.OpenForListing(kmc_prefix)) {
        throw std::runtime_error("Failed to open KMC database for listing: " + kmc_prefix);
    }
    CKMCFileInfo info{};
    if (!kmc.Info(info)) {
        kmc.Close();
        throw std::runtime_error("Failed to read KMC database info: " + kmc_prefix);
    }

    const std::string npy_path = out_prefix + ".npy";
    const std::string id_path = out_prefix + ".id";
    const std::string site_path = out_prefix + ".site.tsv";

    std::ofstream npy(npy_path, std::ios::binary);
    if (!npy.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + npy_path);
    }
    write_npy_f32_header(npy, info.total_kmers, 1);

    std::ofstream idf(id_path, std::ios::out | std::ios::trunc);
    if (!idf.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + id_path);
    }
    idf << sample_id << "\n";
    idf.flush();
    if (!idf.good()) {
        kmc.Close();
        throw std::runtime_error("Failed writing ID file: " + id_path);
    }

    std::ofstream site(site_path, std::ios::out | std::ios::trunc);
    if (!site.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + site_path);
    }
    site << "#CHROM\tPOS\tREF\tALT\tcount\n";

    CKmerAPI kmer(info.kmer_length);
    std::uint64_t count = 0;
    std::uint64_t written = 0;
    while (kmc.ReadNextKmer(kmer, count)) {
        const float c = static_cast<float>(count);
        npy.write(reinterpret_cast<const char*>(&c), sizeof(float));
        if (!npy.good()) {
            kmc.Close();
            throw std::runtime_error("Failed writing NPY payload: " + npy_path);
        }
        const std::string kmer_str = kmer.to_string();
        site << "KMER\t" << (written + 1) << "\tN\t" << kmer_str << "\t" << count << "\n";
        if (!site.good()) {
            kmc.Close();
            throw std::runtime_error("Failed writing site file: " + site_path);
        }
        ++written;
    }
    kmc.Close();

    npy.flush();
    site.flush();
    if (!npy.good()) {
        throw std::runtime_error("Failed finalizing NPY output: " + npy_path);
    }
    if (!site.good()) {
        throw std::runtime_error("Failed finalizing site output: " + site_path);
    }

    py::dict out;
    out["n_kmers"] = written;
    out["kmer_length"] = info.kmer_length;
    out["sample_id"] = sample_id;
    out["npy"] = npy_path;
    out["id"] = id_path;
    out["site"] = site_path;
    return out;
}

static py::dict kmc_export_bin_single(
    const std::string& kmc_prefix,
    const std::string& out_prefix,
    const std::string& sample_id = "sample",
    py::object progress_callback = py::none(),
    std::uint64_t progress_every = 200000,
    std::uint32_t threads = 0
) {
    if (kmc_prefix.empty()) {
        throw std::runtime_error("kmc_prefix cannot be empty.");
    }
    if (out_prefix.empty()) {
        throw std::runtime_error("out_prefix cannot be empty.");
    }
    if (sample_id.empty()) {
        throw std::runtime_error("sample_id cannot be empty.");
    }

    CKMCFile kmc;
    if (!kmc.OpenForListing(kmc_prefix)) {
        throw std::runtime_error("Failed to open KMC database for listing: " + kmc_prefix);
    }
    CKMCFileInfo info{};
    if (!kmc.Info(info)) {
        kmc.Close();
        throw std::runtime_error("Failed to read KMC database info: " + kmc_prefix);
    }

    const std::string bin_path = out_prefix + ".bin";
    const std::string id_path = out_prefix + ".bin.id";
    const std::string site_path = out_prefix + ".bin.site";

    std::ofstream binf(bin_path, std::ios::binary | std::ios::trunc);
    if (!binf.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + bin_path);
    }
    std::ofstream sitef(site_path, std::ios::binary | std::ios::trunc);
    if (!sitef.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + site_path);
    }
    std::ofstream idf(id_path, std::ios::out | std::ios::trunc);
    if (!idf.is_open()) {
        kmc.Close();
        throw std::runtime_error("Failed to open output file: " + id_path);
    }
    idf << sample_id << "\n";
    idf.flush();
    if (!idf.good()) {
        kmc.Close();
        throw std::runtime_error("Failed writing ID file: " + id_path);
    }

    // Placeholder headers; n_sites will be patched after enumeration.
    write_bin01_header(binf, 0ULL, 1ULL);
    write_bin_site_header(sitef, 0ULL);
    const bool has_progress = !progress_callback.is_none();
    const std::uint64_t progress_step = (progress_every > 0) ? progress_every : 200000ULL;
    std::uint64_t next_progress_emit = progress_step;
    const std::uint64_t total_input_records = static_cast<std::uint64_t>(info.total_kmers);
    if (has_progress) {
        progress_callback(0ULL, 0ULL, total_input_records);
    }

    CKmerAPI kmer(info.kmer_length);
    std::uint64_t count = 0;
    std::uint64_t written = 0;
    const std::uint8_t present_byte = 0x01u; // single-sample presence row
    const std::uint32_t encode_threads = normalize_threads(threads);
    constexpr std::size_t encode_batch_size = 8192;
    std::vector<std::string> pending_kmers;
    pending_kmers.reserve(encode_batch_size);
    std::vector<std::vector<std::uint8_t>> pending_encoded;
    auto flush_pending = [&]() {
        if (pending_kmers.empty()) {
            return;
        }
        encode_site_records_parallel(pending_kmers, pending_encoded, encode_threads);
        for (std::size_t i = 0; i < pending_kmers.size(); ++i) {
            binf.write(reinterpret_cast<const char*>(&present_byte), 1);
            if (!binf.good()) {
                kmc.Close();
                throw std::runtime_error("Failed writing BIN payload: " + bin_path);
            }
            const std::vector<std::uint8_t>& rec = pending_encoded[i];
            sitef.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(rec.size()));
            if (!sitef.good()) {
                kmc.Close();
                throw std::runtime_error("Failed writing BIN site payload: " + site_path);
            }
        }
        pending_kmers.clear();
        pending_encoded.clear();
    };
    while (kmc.ReadNextKmer(kmer, count)) {
        pending_kmers.push_back(kmer.to_string());
        if (pending_kmers.size() >= encode_batch_size) {
            flush_pending();
        }
        ++written;
        if (has_progress && written >= next_progress_emit) {
            progress_callback(written, written, total_input_records);
            next_progress_emit = written + progress_step;
        }
    }
    flush_pending();
    kmc.Close();

    // Patch final site counts in headers.
    write_bin01_header(binf, written, 1ULL);
    write_bin_site_header(sitef, written);

    binf.flush();
    sitef.flush();
    if (!binf.good()) {
        throw std::runtime_error("Failed finalizing BIN output: " + bin_path);
    }
    if (!sitef.good()) {
        throw std::runtime_error("Failed finalizing BIN site output: " + site_path);
    }

    py::dict out;
    out["n_kmers"] = written;
    out["kmer_length"] = info.kmer_length;
    out["sample_id"] = sample_id;
    out["processed_records"] = written;
    out["total_input_records"] = total_input_records;
    out["encode_threads"] = encode_threads;
    out["bin"] = bin_path;
    out["id"] = id_path;
    out["site"] = site_path;
    if (has_progress) {
        progress_callback(written, written, total_input_records);
    }
    return out;
}

static py::dict kmc_export_bin_multi(
    const std::vector<std::string>& kmc_prefixes,
    const std::string& out_prefix,
    const std::vector<std::string>& sample_ids,
    std::uint64_t max_kmers = 0,
    double kmerf = 0.2,
    py::object progress_callback = py::none(),
    std::uint64_t progress_every = 200000,
    py::object benchmark_callback = py::none(),
    std::uint64_t benchmark_progress_every = 5000,
    double benchmark_fraction = 0.01,
    std::uint32_t threads = 0
) {
    if (kmc_prefixes.empty()) {
        throw std::runtime_error("kmc_prefixes cannot be empty.");
    }
    if (out_prefix.empty()) {
        throw std::runtime_error("out_prefix cannot be empty.");
    }
    if (sample_ids.size() != kmc_prefixes.size()) {
        throw std::runtime_error("sample_ids length must match kmc_prefixes length.");
    }
    if (!std::isfinite(kmerf) || kmerf < 0.0 || kmerf > 0.5) {
        throw std::runtime_error("kmerf must be within [0, 0.5].");
    }
    for (const auto& sid : sample_ids) {
        if (sid.empty()) {
            throw std::runtime_error("sample_id cannot be empty.");
        }
    }

    enum class MergeStrategy {
        TwoWayDirect = 0,
        LinearScan = 1,
        LoserTree = 2,
    };
    auto strategy_name = [](MergeStrategy s) -> const char* {
        switch (s) {
            case MergeStrategy::TwoWayDirect:
                return "two_way_direct";
            case MergeStrategy::LinearScan:
                return "linear_scan";
            case MergeStrategy::LoserTree:
                return "loser_tree";
            default:
                return "loser_tree";
        }
    };

    struct Cursor {
        CKMCFile kmc;
        CKMCFileInfo info{};
        CKmerAPI kmer;
        std::uint64_t count = 0;
        std::uint64_t loaded = 0;
        bool active = false;
    };

    auto build_cursors =
        [&]() -> std::tuple<std::vector<std::unique_ptr<Cursor>>, std::uint32_t, std::uint64_t> {
            std::vector<std::unique_ptr<Cursor>> out;
            out.reserve(kmc_prefixes.size());
            std::uint32_t out_k = 0;
            std::uint64_t out_total = 0;
            for (std::size_t i = 0; i < kmc_prefixes.size(); ++i) {
                const std::string& prefix = kmc_prefixes[i];
                if (prefix.empty()) {
                    throw std::runtime_error("kmc_prefix cannot be empty.");
                }
                std::unique_ptr<Cursor> cur(new Cursor());
                if (!cur->kmc.OpenForListing(prefix)) {
                    throw std::runtime_error("Failed to open KMC database for listing: " + prefix);
                }
                if (!cur->kmc.Info(cur->info)) {
                    cur->kmc.Close();
                    throw std::runtime_error("Failed to read KMC database info: " + prefix);
                }
                if (i == 0) {
                    out_k = cur->info.kmer_length;
                } else if (cur->info.kmer_length != out_k) {
                    cur->kmc.Close();
                    throw std::runtime_error(
                        "All KMC databases must have identical k-mer length. Got mismatch at prefix: " + prefix
                    );
                }
                const std::uint64_t this_total = static_cast<std::uint64_t>(cur->info.total_kmers);
                const std::uint64_t this_limit = (max_kmers > 0) ? (std::min)(this_total, max_kmers) : this_total;
                out_total += this_limit;
                cur->kmer = CKmerAPI(cur->info.kmer_length);
                std::uint64_t c = 0;
                if (cur->kmc.ReadNextKmer(cur->kmer, c)) {
                    cur->count = c;
                    cur->loaded = 1;
                    cur->active = true;
                } else {
                    cur->count = 0;
                    cur->loaded = 0;
                    cur->active = false;
                }
                out.push_back(std::move(cur));
            }
            return std::make_tuple(std::move(out), out_k, out_total);
        };

    std::uint32_t common_k = 0;
    std::uint64_t total_input_records = 0;
    std::vector<std::unique_ptr<Cursor>> cursors;
    {
        auto init = build_cursors();
        cursors = std::move(std::get<0>(init));
        common_k = std::get<1>(init);
        total_input_records = std::get<2>(init);
    }

    const std::string bin_path = out_prefix + ".bin";
    const std::string id_path = out_prefix + ".bin.id";
    const std::string site_path = out_prefix + ".bin.site";
    const std::uint64_t n_samples = static_cast<std::uint64_t>(sample_ids.size());
    const std::size_t row_nbytes = static_cast<std::size_t>((n_samples + 7ULL) / 8ULL);
    const double keep_min_ratio = kmerf / 10.0;
    const double keep_max_ratio = 1.0 - keep_min_ratio;
    if (row_nbytes == 0) {
        throw std::runtime_error("Invalid row byte width: zero.");
    }

    std::ofstream binf(bin_path, std::ios::binary | std::ios::trunc);
    if (!binf.is_open()) {
        throw std::runtime_error("Failed to open output file: " + bin_path);
    }
    std::ofstream sitef(site_path, std::ios::binary | std::ios::trunc);
    if (!sitef.is_open()) {
        throw std::runtime_error("Failed to open output file: " + site_path);
    }
    std::ofstream idf(id_path, std::ios::out | std::ios::trunc);
    if (!idf.is_open()) {
        throw std::runtime_error("Failed to open output file: " + id_path);
    }
    for (const auto& sid : sample_ids) {
        idf << sid << "\n";
        if (!idf.good()) {
            throw std::runtime_error("Failed writing ID file: " + id_path);
        }
    }
    idf.flush();
    if (!idf.good()) {
        throw std::runtime_error("Failed finalizing ID file: " + id_path);
    }

    // Placeholder headers; n_sites will be patched after merge.
    write_bin01_header(binf, 0ULL, n_samples);
    write_bin_site_header(sitef, 0ULL);
    const bool has_progress = !progress_callback.is_none();
    const std::uint64_t progress_step = (progress_every > 0) ? progress_every : 200000ULL;
    const bool has_benchmark_progress = !benchmark_callback.is_none();
    const std::uint64_t benchmark_step = (benchmark_progress_every > 0) ? benchmark_progress_every : 5000ULL;
    std::uint64_t processed_records = 0;
    for (std::size_t i = 0; i < cursors.size(); ++i) {
        if (cursors[i]->active) {
            processed_records += 1ULL;
        }
    }
    std::uint64_t next_progress_emit = progress_step;
    if (has_progress) {
        progress_callback(processed_records, 0ULL, total_input_records);
    }

    const double benchmark_fraction_clamped = (std::max)(0.0, (std::min)(0.10, benchmark_fraction));
    std::uint64_t benchmark_budget_total = 0;
    if (benchmark_fraction_clamped > 0.0 && total_input_records > 0ULL) {
        const double raw = static_cast<double>(total_input_records) * benchmark_fraction_clamped;
        benchmark_budget_total = static_cast<std::uint64_t>(raw);
        if (benchmark_budget_total == 0ULL) {
            benchmark_budget_total = 1ULL;
        }
        benchmark_budget_total = (std::min)(benchmark_budget_total, total_input_records);
    }
    std::uint64_t benchmark_records_done = 0ULL;
    std::uint64_t benchmark_next_emit = benchmark_step;

    auto emit_benchmark_progress = [&](bool force, std::uint64_t status_code) {
        if (!has_benchmark_progress || benchmark_budget_total == 0ULL) {
            return;
        }
        if (force || benchmark_records_done >= benchmark_next_emit) {
            benchmark_callback(benchmark_records_done, benchmark_budget_total, status_code);
            benchmark_next_emit = benchmark_records_done + benchmark_step;
        }
    };

    auto advance_cursor = [&](std::size_t idx) {
        Cursor& cur = *cursors[idx];
        if (!cur.active) {
            return;
        }
        if (max_kmers > 0 && cur.loaded >= max_kmers) {
            cur.active = false;
            return;
        }
        std::uint64_t c = 0;
        if (cur.kmc.ReadNextKmer(cur.kmer, c)) {
            cur.count = c;
            cur.loaded += 1;
            cur.active = true;
            processed_records += 1ULL;
        } else {
            cur.active = false;
        }
    };

    auto close_all = [&](std::vector<std::unique_ptr<Cursor>>& cset) {
        for (auto& cur : cset) {
            cur->kmc.Close();
        }
    };

    const std::size_t loser_npos = std::numeric_limits<std::size_t>::max();
    auto loser_pick = [&](auto& cset, std::size_t a, std::size_t b) -> std::pair<std::size_t, std::size_t> {
        auto is_active = [&](std::size_t idx) -> bool {
            return idx != loser_npos && idx < cset.size() && cset[idx]->active;
        };
        const bool a_active = is_active(a);
        const bool b_active = is_active(b);
        if (!a_active && !b_active) {
            return std::make_pair(loser_npos, loser_npos);
        }
        if (!a_active) {
            return std::make_pair(b, loser_npos);
        }
        if (!b_active) {
            return std::make_pair(a, loser_npos);
        }
        if (cset[a]->kmer < cset[b]->kmer) {
            return std::make_pair(a, b);
        }
        if (cset[b]->kmer < cset[a]->kmer) {
            return std::make_pair(b, a);
        }
        if (a <= b) {
            return std::make_pair(a, b);
        }
        return std::make_pair(b, a);
    };
    auto loser_init = [&](auto& cset,
                          std::vector<std::size_t>& winners,
                          std::vector<std::size_t>& losers,
                          std::size_t& leaf_base) {
        const std::size_t n = cset.size();
        leaf_base = 1;
        while (leaf_base < n) {
            leaf_base <<= 1U;
        }
        winners.assign(leaf_base * 2U, loser_npos);
        losers.assign(leaf_base, loser_npos);
        for (std::size_t i = 0; i < n; ++i) {
            winners[leaf_base + i] = cset[i]->active ? i : loser_npos;
        }
        for (std::size_t node = leaf_base; node-- > 1U;) {
            const auto wl = loser_pick(cset, winners[node << 1U], winners[(node << 1U) | 1U]);
            winners[node] = wl.first;
            losers[node] = wl.second;
        }
    };
    auto loser_update = [&](auto& cset,
                            std::vector<std::size_t>& winners,
                            std::vector<std::size_t>& losers,
                            std::size_t leaf_base,
                            std::size_t idx) {
        std::size_t node = leaf_base + idx;
        winners[node] = cset[idx]->active ? idx : loser_npos;
        node >>= 1U;
        while (node > 0U) {
            const auto wl = loser_pick(cset, winners[node << 1U], winners[(node << 1U) | 1U]);
            winners[node] = wl.first;
            losers[node] = wl.second;
            node >>= 1U;
        }
    };
    auto loser_top = [&](const std::vector<std::size_t>& winners) -> std::size_t {
        if (winners.size() <= 1U) {
            return loser_npos;
        }
        return winners[1];
    };

    const std::uint32_t encode_threads = normalize_threads(threads);
    constexpr std::size_t encode_batch_size = 8192;
    constexpr std::size_t pipeline_queue_cap = 4;
    struct RawBatch {
        std::vector<std::string> kmers;
        std::vector<std::uint8_t> rows;
    };
    struct EncodedBatch {
        std::vector<std::vector<std::uint8_t>> site_records;
        std::vector<std::uint8_t> rows;
    };
    std::deque<RawBatch> raw_queue;
    std::deque<EncodedBatch> encoded_queue;
    std::mutex raw_mtx;
    std::mutex encoded_mtx;
    std::mutex pipe_err_mtx;
    std::condition_variable raw_cv_not_empty;
    std::condition_variable raw_cv_not_full;
    std::condition_variable encoded_cv_not_empty;
    std::condition_variable encoded_cv_not_full;
    bool raw_done = false;
    bool encoded_done = false;
    bool pipeline_abort = false;
    std::exception_ptr pipeline_eptr = nullptr;

    auto set_pipeline_error = [&](std::exception_ptr eptr) {
        {
            std::lock_guard<std::mutex> lg(pipe_err_mtx);
            if (!pipeline_eptr) {
                pipeline_eptr = eptr;
            }
        }
        {
            std::lock_guard<std::mutex> lk_raw(raw_mtx);
            pipeline_abort = true;
            raw_done = true;
        }
        {
            std::lock_guard<std::mutex> lk_enc(encoded_mtx);
            encoded_done = true;
        }
        raw_cv_not_empty.notify_all();
        raw_cv_not_full.notify_all();
        encoded_cv_not_empty.notify_all();
        encoded_cv_not_full.notify_all();
    };
    auto rethrow_pipeline_error = [&]() {
        std::exception_ptr eptr_local = nullptr;
        {
            std::lock_guard<std::mutex> lg(pipe_err_mtx);
            eptr_local = pipeline_eptr;
        }
        if (eptr_local) {
            std::rethrow_exception(eptr_local);
        }
    };

    std::vector<std::size_t> equal_indices;
    equal_indices.reserve(cursors.size());
    RawBatch pending_batch;
    pending_batch.kmers.reserve(encode_batch_size);
    pending_batch.rows.reserve(encode_batch_size * row_nbytes);

    auto enqueue_pending_batch = [&]() {
        if (pending_batch.kmers.empty()) {
            return;
        }
        {
            std::unique_lock<std::mutex> lk(raw_mtx);
            raw_cv_not_full.wait(lk, [&]() {
                return pipeline_abort || raw_queue.size() < pipeline_queue_cap;
            });
            if (pipeline_abort) {
                lk.unlock();
                rethrow_pipeline_error();
                throw std::runtime_error("kmerge pipeline aborted.");
            }
            raw_queue.push_back(std::move(pending_batch));
        }
        raw_cv_not_empty.notify_one();
        pending_batch = RawBatch{};
        pending_batch.kmers.reserve(encode_batch_size);
        pending_batch.rows.reserve(encode_batch_size * row_nbytes);
    };

    std::thread encoder_thread;
    std::thread writer_thread;
    auto join_pipeline_threads = [&]() {
        if (encoder_thread.joinable()) {
            encoder_thread.join();
        }
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
    };
    auto start_pipeline_threads = [&]() {
        encoder_thread = std::thread([&]() {
            try {
                while (true) {
                    RawBatch raw_batch;
                    {
                        std::unique_lock<std::mutex> lk(raw_mtx);
                        raw_cv_not_empty.wait(lk, [&]() {
                            return pipeline_abort || !raw_queue.empty() || raw_done;
                        });
                        if ((pipeline_abort && raw_queue.empty()) || (raw_queue.empty() && raw_done)) {
                            break;
                        }
                        if (raw_queue.empty()) {
                            continue;
                        }
                        raw_batch = std::move(raw_queue.front());
                        raw_queue.pop_front();
                    }
                    raw_cv_not_full.notify_one();

                    EncodedBatch encoded_batch;
                    encoded_batch.rows = std::move(raw_batch.rows);
                    encode_site_records_parallel(raw_batch.kmers, encoded_batch.site_records, encode_threads);

                    {
                        std::unique_lock<std::mutex> lk(encoded_mtx);
                        encoded_cv_not_full.wait(lk, [&]() {
                            return pipeline_abort || encoded_queue.size() < pipeline_queue_cap;
                        });
                        if (pipeline_abort) {
                            break;
                        }
                        encoded_queue.push_back(std::move(encoded_batch));
                    }
                    encoded_cv_not_empty.notify_one();
                }
            } catch (...) {
                set_pipeline_error(std::current_exception());
            }
            {
                std::lock_guard<std::mutex> lk(encoded_mtx);
                encoded_done = true;
            }
            encoded_cv_not_empty.notify_all();
        });

        writer_thread = std::thread([&]() {
            try {
                while (true) {
                    EncodedBatch encoded_batch;
                    {
                        std::unique_lock<std::mutex> lk(encoded_mtx);
                        encoded_cv_not_empty.wait(lk, [&]() {
                            return pipeline_abort || !encoded_queue.empty() || encoded_done;
                        });
                        if ((pipeline_abort && encoded_queue.empty()) || (encoded_queue.empty() && encoded_done)) {
                            break;
                        }
                        if (encoded_queue.empty()) {
                            continue;
                        }
                        encoded_batch = std::move(encoded_queue.front());
                        encoded_queue.pop_front();
                    }
                    encoded_cv_not_full.notify_one();

                    const std::size_t n = encoded_batch.site_records.size();
                    if (encoded_batch.rows.size() != n * row_nbytes) {
                        throw std::runtime_error("Internal row payload size mismatch in kmerge pipeline.");
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        const std::size_t off = i * row_nbytes;
                        binf.write(
                            reinterpret_cast<const char*>(encoded_batch.rows.data() + off),
                            static_cast<std::streamsize>(row_nbytes)
                        );
                        if (!binf.good()) {
                            throw std::runtime_error("Failed writing BIN payload: " + bin_path);
                        }
                        const std::vector<std::uint8_t>& rec = encoded_batch.site_records[i];
                        sitef.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(rec.size()));
                        if (!sitef.good()) {
                            throw std::runtime_error("Failed writing BIN site payload: " + site_path);
                        }
                    }
                }
            } catch (...) {
                set_pipeline_error(std::current_exception());
            }
        });
    };

    auto benchmark_strategy = [&](MergeStrategy strategy, std::uint64_t strategy_budget) -> double {
        if (strategy_budget == 0ULL) {
            return std::numeric_limits<double>::infinity();
        }
        std::vector<std::unique_ptr<Cursor>> bench_cursors;
        std::uint32_t bench_k = 0;
        std::uint64_t bench_total = 0;
        {
            auto bench_init = build_cursors();
            bench_cursors = std::move(std::get<0>(bench_init));
            bench_k = std::get<1>(bench_init);
            bench_total = std::get<2>(bench_init);
        }
        (void)bench_k;
        if (bench_total == 0) {
            close_all(bench_cursors);
            return std::numeric_limits<double>::infinity();
        }

        auto advance_bench = [&](std::size_t idx) {
            Cursor& cur = *bench_cursors[idx];
            if (!cur.active) {
                return;
            }
            if (max_kmers > 0 && cur.loaded >= max_kmers) {
                cur.active = false;
                return;
            }
            std::uint64_t c = 0;
            if (cur.kmc.ReadNextKmer(cur.kmer, c)) {
                cur.count = c;
                cur.loaded += 1;
                cur.active = true;
            } else {
                cur.active = false;
            }
        };

        const std::uint64_t warmup_len = (std::max)(1ULL, strategy_budget / 10ULL);
        const std::uint64_t remaining = (strategy_budget > warmup_len) ? (strategy_budget - warmup_len) : 1ULL;
        const std::uint64_t unit = (std::max)(1ULL, remaining / 5ULL);
        const std::uint64_t window_len = unit;  // front/mid/back windows
        const std::uint64_t skip_len = unit;    // gaps between windows
        const std::uint64_t n_windows = 3ULL;

        auto consume_with_budget = [&](auto&& consume_one, std::uint64_t target_records) -> std::uint64_t {
            std::uint64_t consumed = 0ULL;
            while (consumed < target_records) {
                const std::uint64_t step = consume_one();
                if (step == 0ULL) {
                    break;
                }
                consumed += step;
                if (benchmark_budget_total > 0ULL) {
                    const std::uint64_t room = (benchmark_records_done < benchmark_budget_total)
                        ? (benchmark_budget_total - benchmark_records_done)
                        : 0ULL;
                    const std::uint64_t add = (std::min)(room, step);
                    benchmark_records_done += add;
                    emit_benchmark_progress(false, 0ULL);
                }
            }
            return consumed;
        };

        auto run_windowed = [&](auto&& consume_one) -> double {
            const std::uint64_t warmup_done = consume_with_budget(consume_one, warmup_len);
            if (warmup_done == 0ULL) {
                close_all(bench_cursors);
                return std::numeric_limits<double>::infinity();
            }
            std::vector<double> ns_per_record;
            ns_per_record.reserve(static_cast<std::size_t>(n_windows));
            for (std::uint64_t w = 0; w < n_windows; ++w) {
                const auto t0 = std::chrono::steady_clock::now();
                const std::uint64_t measured = consume_with_budget(consume_one, window_len);
                const auto t1 = std::chrono::steady_clock::now();
                if (measured == 0ULL) {
                    break;
                }
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                ns_per_record.push_back(static_cast<double>(ns) / static_cast<double>(measured));
                if (w + 1 < n_windows) {
                    const std::uint64_t skipped = consume_with_budget(consume_one, skip_len);
                    if (skipped == 0ULL) {
                        break;
                    }
                }
            }
            close_all(bench_cursors);
            if (ns_per_record.empty()) {
                return std::numeric_limits<double>::infinity();
            }
            std::sort(ns_per_record.begin(), ns_per_record.end());
            return ns_per_record[ns_per_record.size() / 2];
        };

        if (strategy == MergeStrategy::LinearScan) {
            auto consume_one_linear = [&]() -> std::uint64_t {
                std::size_t min_idx = std::numeric_limits<std::size_t>::max();
                for (std::size_t i = 0; i < bench_cursors.size(); ++i) {
                    if (!bench_cursors[i]->active) {
                        continue;
                    }
                    if (
                        min_idx == std::numeric_limits<std::size_t>::max() ||
                        bench_cursors[i]->kmer < bench_cursors[min_idx]->kmer
                    ) {
                        min_idx = i;
                    }
                }
                if (min_idx == std::numeric_limits<std::size_t>::max()) {
                    return 0ULL;
                }
                const CKmerAPI& min_kmer = bench_cursors[min_idx]->kmer;
                equal_indices.clear();
                for (std::size_t i = 0; i < bench_cursors.size(); ++i) {
                    if (bench_cursors[i]->active && bench_cursors[i]->kmer == min_kmer) {
                        equal_indices.push_back(i);
                    }
                }
                for (std::size_t idx : equal_indices) {
                    advance_bench(idx);
                }
                return static_cast<std::uint64_t>(equal_indices.size());
            };
            return run_windowed(consume_one_linear);
        }

        std::vector<std::size_t> bench_winners;
        std::vector<std::size_t> bench_losers;
        std::size_t bench_leaf_base = 1U;
        loser_init(bench_cursors, bench_winners, bench_losers, bench_leaf_base);
        auto consume_one_loser_tree = [&]() -> std::uint64_t {
            const std::size_t first_idx = loser_top(bench_winners);
            if (first_idx == loser_npos) {
                return 0ULL;
            }
            const CKmerAPI min_kmer = bench_cursors[first_idx]->kmer;
            equal_indices.clear();
            while (true) {
                const std::size_t idx = loser_top(bench_winners);
                if (idx == loser_npos || !bench_cursors[idx]->active || !(bench_cursors[idx]->kmer == min_kmer)) {
                    break;
                }
                equal_indices.push_back(idx);
                advance_bench(idx);
                loser_update(bench_cursors, bench_winners, bench_losers, bench_leaf_base, idx);
            }
            return static_cast<std::uint64_t>(equal_indices.size());
        };
        return run_windowed(consume_one_loser_tree);
    };

    MergeStrategy selected_strategy = MergeStrategy::LoserTree;
    double bench_linear_ns = std::numeric_limits<double>::infinity();
    double bench_loser_tree_ns = std::numeric_limits<double>::infinity();
    const std::size_t n_way = cursors.size();
    if (n_way == 2) {
        selected_strategy = MergeStrategy::TwoWayDirect;
    } else if (n_way >= 3 && n_way <= 8) {
        if (benchmark_budget_total > 0ULL) {
            if (has_benchmark_progress) {
                benchmark_callback(0ULL, benchmark_budget_total, 0ULL);
            }
            const std::uint64_t per_strategy_budget = (std::max)(1ULL, benchmark_budget_total / 2ULL);
            bench_linear_ns = benchmark_strategy(MergeStrategy::LinearScan, per_strategy_budget);
            bench_loser_tree_ns = benchmark_strategy(MergeStrategy::LoserTree, per_strategy_budget);
            constexpr double switch_threshold = 0.10; // Require >=10% win to switch.
            if (std::isfinite(bench_linear_ns) && std::isfinite(bench_loser_tree_ns)) {
                if (bench_linear_ns <= bench_loser_tree_ns * (1.0 - switch_threshold)) {
                    selected_strategy = MergeStrategy::LinearScan;
                } else if (bench_loser_tree_ns <= bench_linear_ns * (1.0 - switch_threshold)) {
                    selected_strategy = MergeStrategy::LoserTree;
                } else {
                    selected_strategy = MergeStrategy::LoserTree;
                }
            } else if (std::isfinite(bench_linear_ns)) {
                selected_strategy = MergeStrategy::LinearScan;
            } else {
                selected_strategy = MergeStrategy::LoserTree;
            }
            benchmark_records_done = benchmark_budget_total;
            std::uint64_t strategy_code = 0ULL;
            if (selected_strategy == MergeStrategy::LinearScan) {
                strategy_code = 1ULL;
            } else if (selected_strategy == MergeStrategy::LoserTree) {
                strategy_code = 2ULL;
            } else if (selected_strategy == MergeStrategy::TwoWayDirect) {
                strategy_code = 3ULL;
            }
            emit_benchmark_progress(true, strategy_code);
        } else {
            selected_strategy = MergeStrategy::LoserTree;
        }
    } else {
        selected_strategy = MergeStrategy::LoserTree;
    }

    std::uint64_t written = 0;
    std::uint64_t filtered_by_kmerf = 0;
    auto emit_progress_if_needed = [&]() {
        if (has_progress && processed_records >= next_progress_emit) {
            progress_callback(processed_records, written, total_input_records);
            next_progress_emit = processed_records + progress_step;
        }
    };
    start_pipeline_threads();
    auto append_row = [&](const CKmerAPI& min_kmer, const std::vector<std::size_t>& indices) {
        if (pipeline_abort) {
            rethrow_pipeline_error();
            throw std::runtime_error("kmerge pipeline aborted.");
        }
        const double present_ratio = static_cast<double>(indices.size()) / static_cast<double>(n_samples);
        if (present_ratio < keep_min_ratio || present_ratio > keep_max_ratio) {
            ++filtered_by_kmerf;
            return;
        }
        pending_batch.kmers.push_back(min_kmer.to_string());
        const std::size_t old = pending_batch.rows.size();
        pending_batch.rows.resize(old + row_nbytes, static_cast<std::uint8_t>(0));
        std::uint8_t* row_ptr = pending_batch.rows.data() + old;
        for (std::size_t idx : indices) {
            row_ptr[idx >> 3] = static_cast<std::uint8_t>(row_ptr[idx >> 3] | (1u << (idx & 7)));
        }
        ++written;
        if (pending_batch.kmers.size() >= encode_batch_size) {
            enqueue_pending_batch();
        }
    };

    bool cursors_closed = false;
    auto close_cursors_once = [&]() {
        if (!cursors_closed) {
            close_all(cursors);
            cursors_closed = true;
        }
    };
    try {
        if (selected_strategy == MergeStrategy::TwoWayDirect) {
            Cursor& c0 = *cursors[0];
            Cursor& c1 = *cursors[1];
            while (c0.active || c1.active) {
                equal_indices.clear();
                const CKmerAPI* min_kmer = nullptr;
                if (c0.active && c1.active) {
                    if (c0.kmer < c1.kmer) {
                        min_kmer = &c0.kmer;
                        equal_indices.push_back(0);
                    } else if (c1.kmer < c0.kmer) {
                        min_kmer = &c1.kmer;
                        equal_indices.push_back(1);
                    } else {
                        min_kmer = &c0.kmer;
                        equal_indices.push_back(0);
                        equal_indices.push_back(1);
                    }
                } else if (c0.active) {
                    min_kmer = &c0.kmer;
                    equal_indices.push_back(0);
                } else {
                    min_kmer = &c1.kmer;
                    equal_indices.push_back(1);
                }
                append_row(*min_kmer, equal_indices);
                for (std::size_t idx : equal_indices) {
                    advance_cursor(idx);
                }
                emit_progress_if_needed();
            }
        } else if (selected_strategy == MergeStrategy::LinearScan) {
            while (true) {
                std::size_t min_idx = std::numeric_limits<std::size_t>::max();
                for (std::size_t i = 0; i < cursors.size(); ++i) {
                    if (!cursors[i]->active) {
                        continue;
                    }
                    if (
                        min_idx == std::numeric_limits<std::size_t>::max() ||
                        cursors[i]->kmer < cursors[min_idx]->kmer
                    ) {
                        min_idx = i;
                    }
                }
                if (min_idx == std::numeric_limits<std::size_t>::max()) {
                    break;
                }
                const CKmerAPI& min_kmer = cursors[min_idx]->kmer;
                equal_indices.clear();
                for (std::size_t i = 0; i < cursors.size(); ++i) {
                    if (cursors[i]->active && cursors[i]->kmer == min_kmer) {
                        equal_indices.push_back(i);
                    }
                }
                append_row(min_kmer, equal_indices);
                for (std::size_t idx : equal_indices) {
                    advance_cursor(idx);
                }
                emit_progress_if_needed();
            }
        } else {
            std::vector<std::size_t> loser_winners;
            std::vector<std::size_t> loser_losers;
            std::size_t loser_leaf_base = 1U;
            loser_init(cursors, loser_winners, loser_losers, loser_leaf_base);
            while (true) {
                const std::size_t first_idx = loser_top(loser_winners);
                if (first_idx == loser_npos) {
                    break;
                }
                const CKmerAPI min_kmer = cursors[first_idx]->kmer;
                equal_indices.clear();
                while (true) {
                    const std::size_t idx = loser_top(loser_winners);
                    if (idx == loser_npos || !cursors[idx]->active || !(cursors[idx]->kmer == min_kmer)) {
                        break;
                    }
                    equal_indices.push_back(idx);
                    advance_cursor(idx);
                    loser_update(cursors, loser_winners, loser_losers, loser_leaf_base, idx);
                }
                append_row(min_kmer, equal_indices);
                emit_progress_if_needed();
            }
        }

        enqueue_pending_batch();
        {
            std::lock_guard<std::mutex> lk(raw_mtx);
            raw_done = true;
        }
        raw_cv_not_empty.notify_all();
        close_cursors_once();
        join_pipeline_threads();
        rethrow_pipeline_error();
    } catch (...) {
        std::exception_ptr merge_eptr = std::current_exception();
        close_cursors_once();
        {
            std::lock_guard<std::mutex> lk_raw(raw_mtx);
            pipeline_abort = true;
            raw_done = true;
        }
        {
            std::lock_guard<std::mutex> lk_enc(encoded_mtx);
            encoded_done = true;
        }
        raw_cv_not_empty.notify_all();
        raw_cv_not_full.notify_all();
        encoded_cv_not_empty.notify_all();
        encoded_cv_not_full.notify_all();
        join_pipeline_threads();
        try {
            rethrow_pipeline_error();
        } catch (...) {
            throw;
        }
        std::rethrow_exception(merge_eptr);
    }

    // Patch final site counts in headers.
    write_bin01_header(binf, written, n_samples);
    write_bin_site_header(sitef, written);
    binf.flush();
    sitef.flush();
    if (!binf.good()) {
        throw std::runtime_error("Failed finalizing BIN output: " + bin_path);
    }
    if (!sitef.good()) {
        throw std::runtime_error("Failed finalizing BIN site output: " + site_path);
    }

    py::dict out;
    out["n_kmers"] = written;
    out["kmer_length"] = common_k;
    out["n_samples"] = n_samples;
    out["processed_records"] = processed_records;
    out["total_input_records"] = total_input_records;
    out["encode_threads"] = encode_threads;
    out["benchmark_records"] = benchmark_records_done;
    out["benchmark_budget_records"] = benchmark_budget_total;
    out["merge_strategy"] = strategy_name(selected_strategy);
    out["kmerf"] = kmerf;
    out["kmerf_keep_min_ratio"] = keep_min_ratio;
    out["kmerf_keep_max_ratio"] = keep_max_ratio;
    out["filtered_by_kmerf"] = filtered_by_kmerf;
    if (std::isfinite(bench_linear_ns)) {
        out["bench_linear_ns_per_record"] = bench_linear_ns;
    }
    if (std::isfinite(bench_loser_tree_ns)) {
        out["bench_loser_tree_ns_per_record"] = bench_loser_tree_ns;
    }
    out["bin"] = bin_path;
    out["id"] = id_path;
    out["site"] = site_path;
    if (has_progress) {
        progress_callback(processed_records, written, total_input_records);
    }
    return out;
}

static py::dict kmc_dump_pairs(
    const std::string& kmc_prefix,
    std::uint64_t max_kmers = 0
) {
    if (kmc_prefix.empty()) {
        throw std::runtime_error("kmc_prefix cannot be empty.");
    }

    CKMCFile kmc;
    if (!kmc.OpenForListing(kmc_prefix)) {
        throw std::runtime_error("Failed to open KMC database for listing: " + kmc_prefix);
    }
    CKMCFileInfo info{};
    if (!kmc.Info(info)) {
        kmc.Close();
        throw std::runtime_error("Failed to read KMC database info: " + kmc_prefix);
    }

    py::list kmers;
    py::list counts;
    CKmerAPI kmer(info.kmer_length);
    std::uint64_t count = 0;
    std::uint64_t n = 0;
    while (kmc.ReadNextKmer(kmer, count)) {
        kmers.append(py::str(kmer.to_string()));
        counts.append(py::int_(count));
        ++n;
        if (max_kmers > 0 && n >= max_kmers) {
            break;
        }
    }
    kmc.Close();

    py::dict out;
    out["kmer_length"] = info.kmer_length;
    out["n"] = n;
    out["kmers"] = kmers;
    out["counts"] = counts;
    return out;
}

PYBIND11_MODULE(_kmc_count, m) {
    m.doc() = "JanusX KMC counter binding (FASTA/FASTQ -> .kmc_pre/.kmc_suf)";
    m.def(
        "kmc_count",
        &kmc_count,
        py::arg("input_files"),
        py::arg("output_prefix"),
        py::arg("tmp_dir") = ".",
        py::arg("kmer_len") = 31,
        py::arg("threads") = 0,
        py::arg("max_ram_gb") = 12,
        py::arg("cutoff_min") = 2,
        py::arg("cutoff_max") = 1000000000ULL,
        py::arg("counter_max") = 255ULL,
        py::arg("canonical") = true,
        py::arg("input_type") = "fastq"
    );
    m.def("kmc_db_info", &kmc_db_info, py::arg("kmc_prefix"));
    m.def(
        "kmc_export_janusx_single",
        &kmc_export_janusx_single,
        py::arg("kmc_prefix"),
        py::arg("out_prefix"),
        py::arg("sample_id") = "sample"
    );
    m.def(
        "kmc_export_bin_single",
        &kmc_export_bin_single,
        py::arg("kmc_prefix"),
        py::arg("out_prefix"),
        py::arg("sample_id") = "sample",
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 200000,
        py::arg("threads") = 0
    );
    m.def(
        "kmc_export_bin_multi",
        &kmc_export_bin_multi,
        py::arg("kmc_prefixes"),
        py::arg("out_prefix"),
        py::arg("sample_ids"),
        py::arg("max_kmers") = 0,
        py::arg("kmerf") = 0.2,
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 200000,
        py::arg("benchmark_callback") = py::none(),
        py::arg("benchmark_progress_every") = 5000,
        py::arg("benchmark_fraction") = 0.01,
        py::arg("threads") = 0
    );
    m.def(
        "kmc_dump_pairs",
        &kmc_dump_pairs,
        py::arg("kmc_prefix"),
        py::arg("max_kmers") = 0
    );
}
