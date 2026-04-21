#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmc_runner.h"

namespace py = pybind11;

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
}
