#![allow(dead_code)]

use super::score::{score_cont_centered_gain_from_sum_and_n_hit, validate_continuous_y};
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use std::time::Instant;

#[derive(Clone, Debug)]
struct BatchScoreParts {
    raw_score: Vec<f64>,
    mean_hit: Vec<f64>,
    mean_miss: Vec<f64>,
    support_frac: Vec<f64>,
    n_hit: Vec<u32>,
    n_miss: Vec<u32>,
    sum_hit: Vec<f64>,
    total_sum: f64,
    n_rows: usize,
    row_words: usize,
    n_samples: usize,
}

#[inline]
fn validate_batch_shape(
    y: &[f64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    ctx: &str,
) -> Result<(), String> {
    if n_rows == 0 {
        return Err(format!("{ctx}: n_rows must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    let bit_cap = row_words.saturating_mul(64);
    if n_samples > bit_cap {
        return Err(format!(
            "{ctx}: n_samples={} exceeds bit capacity={} (row_words={})",
            n_samples, bit_cap, row_words
        ));
    }
    validate_continuous_y(y, n_samples, ctx)?;
    Ok(())
}

#[inline]
fn count_sum_y_where_bit1_row(bits: &[u64], y: &[f64], n_samples: usize) -> (u32, f64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut n1 = 0u32;
    let mut s1 = 0.0f64;

    for (w_idx, &word) in bits.iter().take(full_words).enumerate() {
        n1 = n1.saturating_add(word.count_ones());
        let mut w = word;
        let base = w_idx << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let mut w = bits[full_words] & mask;
        n1 = n1.saturating_add(w.count_ones());
        let base = full_words << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    (n1, s1)
}

fn build_score_parts_from_sum_hit(
    sum_hit: Vec<f64>,
    n_hit: Vec<u32>,
    n_samples: usize,
    total_sum: f64,
    row_words: usize,
) -> BatchScoreParts {
    let n_rows = sum_hit.len();
    let mut raw_score = Vec::with_capacity(n_rows);
    let mut mean_hit = Vec::with_capacity(n_rows);
    let mut mean_miss = Vec::with_capacity(n_rows);
    let mut support_frac = Vec::with_capacity(n_rows);
    let mut n_miss = Vec::with_capacity(n_rows);
    for (&sum_hit_i, &n_hit_i) in sum_hit.iter().zip(n_hit.iter()) {
        let sc = score_cont_centered_gain_from_sum_and_n_hit(
            total_sum,
            sum_hit_i,
            n_samples,
            n_hit_i as usize,
        );
        raw_score.push(sc.raw_score);
        mean_hit.push(sc.mean_hit);
        mean_miss.push(sc.mean_miss);
        support_frac.push(sc.support_frac);
        n_miss.push(sc.n_miss as u32);
    }
    BatchScoreParts {
        raw_score,
        mean_hit,
        mean_miss,
        support_frac,
        n_hit,
        n_miss,
        sum_hit,
        total_sum,
        n_rows,
        row_words,
        n_samples,
    }
}

fn score_cont_centered_gain_batch_packed_cpu_impl(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<BatchScoreParts, String> {
    const CTX: &str = "garfield_score_cont_centered_gain_batch_packed_cpu";
    validate_batch_shape(y, row_words, n_rows, n_samples, CTX)?;
    if bits_flat.len() != row_words.saturating_mul(n_rows) {
        return Err(format!(
            "{CTX}: bits size mismatch: got {}, expected {} (n_rows={} row_words={})",
            bits_flat.len(),
            row_words.saturating_mul(n_rows),
            n_rows,
            row_words
        ));
    }
    let total_sum = y.iter().take(n_samples).copied().sum::<f64>();
    let mut sum_hit = Vec::with_capacity(n_rows);
    let mut n_hit = Vec::with_capacity(n_rows);
    for rid in 0..n_rows {
        let row = &bits_flat[rid * row_words..(rid + 1) * row_words];
        let (n1, s1) = count_sum_y_where_bit1_row(row, y, n_samples);
        n_hit.push(n1);
        sum_hit.push(s1);
    }
    Ok(build_score_parts_from_sum_hit(
        sum_hit, n_hit, n_samples, total_sum, row_words,
    ))
}

fn batch_score_parts_to_pydict<'py>(
    py: Python<'py>,
    parts: BatchScoreParts,
    backend: &str,
    elapsed_ms: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    out.set_item(
        "raw_score",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.raw_score)).into_bound(),
    )?;
    out.set_item(
        "mean_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.mean_hit)).into_bound(),
    )?;
    out.set_item(
        "mean_miss",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.mean_miss)).into_bound(),
    )?;
    out.set_item(
        "support_frac",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.support_frac)).into_bound(),
    )?;
    out.set_item(
        "n_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.n_hit)).into_bound(),
    )?;
    out.set_item(
        "n_miss",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.n_miss)).into_bound(),
    )?;
    out.set_item(
        "sum_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.sum_hit)).into_bound(),
    )?;
    out.set_item("backend", backend)?;
    out.set_item("total_sum", parts.total_sum)?;
    out.set_item("n_rows", parts.n_rows)?;
    out.set_item("row_words", parts.row_words)?;
    out.set_item("n_samples", parts.n_samples)?;
    if let Some(v) = elapsed_ms {
        out.set_item("elapsed_ms", v)?;
    }
    Ok(out)
}

const GPU_Y_PRECISION_NOTE: &str =
    "gpu path uses f32 accumulation; CPU reference is run on y cast to f32 then lifted to f64";

#[derive(Clone, Debug)]
struct BatchDiffMetrics {
    max_abs_sum_hit_diff: f64,
    mean_abs_sum_hit_diff: f64,
    max_abs_raw_score_diff: f64,
    mean_abs_raw_score_diff: f64,
    max_abs_support_frac_diff: f64,
}

#[derive(Clone, Debug)]
struct GpuRunMeta {
    variant: &'static str,
    y_precision: &'static str,
    actual_threads_per_threadgroup: usize,
    thread_execution_width: usize,
}

fn diff_batch_parts(lhs: &BatchScoreParts, rhs: &BatchScoreParts) -> BatchDiffMetrics {
    let n_rows = lhs.n_rows.min(rhs.n_rows);
    let mut out = BatchDiffMetrics {
        max_abs_sum_hit_diff: 0.0,
        mean_abs_sum_hit_diff: 0.0,
        max_abs_raw_score_diff: 0.0,
        mean_abs_raw_score_diff: 0.0,
        max_abs_support_frac_diff: 0.0,
    };
    for i in 0..n_rows {
        let d_sum = (lhs.sum_hit[i] - rhs.sum_hit[i]).abs();
        let d_raw = (lhs.raw_score[i] - rhs.raw_score[i]).abs();
        let d_frac = (lhs.support_frac[i] - rhs.support_frac[i]).abs();
        out.max_abs_sum_hit_diff = out.max_abs_sum_hit_diff.max(d_sum);
        out.mean_abs_sum_hit_diff += d_sum;
        out.max_abs_raw_score_diff = out.max_abs_raw_score_diff.max(d_raw);
        out.mean_abs_raw_score_diff += d_raw;
        out.max_abs_support_frac_diff = out.max_abs_support_frac_diff.max(d_frac);
    }
    if n_rows > 0 {
        out.mean_abs_sum_hit_diff /= n_rows as f64;
        out.mean_abs_raw_score_diff /= n_rows as f64;
    }
    out
}

fn build_y_word_sums_f32(y: &[f32], row_words: usize, n_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; row_words];
    for (idx, val) in y.iter().take(n_samples).enumerate() {
        out[idx >> 6] += *val;
    }
    out
}

#[pyfunction(name = "garfield_score_cont_centered_gain_batch_packed_cpu")]
pub fn garfield_score_cont_centered_gain_batch_packed_cpu_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let yv = y.as_slice()?;
    let shape = bits.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("bits must be a 2D uint64 array"));
    }
    let n_rows = shape[0];
    let row_words = shape[1];
    let bits_flat = bits
        .as_slice()
        .map_err(|_| PyValueError::new_err("bits must be contiguous (C-order)"))?;
    let t0 = Instant::now();
    let parts = py
        .detach(|| {
            score_cont_centered_gain_batch_packed_cpu_impl(
                yv, bits_flat, row_words, n_rows, n_samples,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    batch_score_parts_to_pydict(py, parts, "cpu", Some(elapsed_ms))
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal_impl {
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize,
    };
    use std::cmp::min;
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::ptr;

    use super::{GpuRunMeta, GPU_Y_PRECISION_NOTE};

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct ScoreParams {
        n_rows: u32,
        row_words: u32,
        n_samples: u32,
        _pad: u32,
    }

    const KERNEL_SRC: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    constant uint JX_MAX_SCORE_TG = 256u;

    struct ScoreParams {
        uint n_rows;
        uint row_words;
        uint n_samples;
        uint _pad;
    };

    kernel void garfield_score_sum_hit_n_hit_batch_legacy(
        device const float* y        [[ buffer(0) ]],
        device const ulong* bits     [[ buffer(1) ]],
        device float* sum_hit_out    [[ buffer(2) ]],
        device uint* n_hit_out       [[ buffer(3) ]],
        constant ScoreParams& prm    [[ buffer(4) ]],
        uint tid                     [[ thread_position_in_grid ]]
    ) {
        if (tid >= prm.n_rows) return;
        uint row_off = tid * prm.row_words;
        uint sample_base = 0u;
        float sum_hit = 0.0f;
        uint n_hit = 0u;
        for (uint w = 0u; w < prm.row_words; ++w) {
            ulong word = bits[row_off + w];
            uint remain = (prm.n_samples > sample_base) ? (prm.n_samples - sample_base) : 0u;
            uint limit = min(64u, remain);
            for (uint b = 0u; b < limit; ++b) {
                if (((word >> b) & 1ul) != 0ul) {
                    sum_hit += y[sample_base + b];
                    n_hit += 1u;
                }
            }
            sample_base += 64u;
            if (sample_base >= prm.n_samples) {
                break;
            }
        }
        sum_hit_out[tid] = sum_hit;
        n_hit_out[tid] = n_hit;
    }

    kernel void garfield_score_sum_hit_n_hit_batch_opt(
        device const float* y             [[ buffer(0) ]],
        device const ulong* bits          [[ buffer(1) ]],
        device const float* y_word_sums   [[ buffer(2) ]],
        device float* sum_hit_out         [[ buffer(3) ]],
        device uint* n_hit_out            [[ buffer(4) ]],
        constant ScoreParams& prm         [[ buffer(5) ]],
        uint lid                          [[ thread_index_in_threadgroup ]],
        uint lsize                        [[ threads_per_threadgroup ]],
        uint group_id                     [[ threadgroup_position_in_grid ]]
    ) {
        if (group_id >= prm.n_rows) return;

        threadgroup float partial_sum[JX_MAX_SCORE_TG];
        threadgroup uint partial_n[JX_MAX_SCORE_TG];

        uint row_off = group_id * prm.row_words;
        float local_sum = 0.0f;
        uint local_n = 0u;

        for (uint w = lid; w < prm.row_words; w += lsize) {
            uint sample_base = w * 64u;
            if (sample_base >= prm.n_samples) {
                break;
            }
            uint remain = prm.n_samples - sample_base;
            uint valid_bits = min(64u, remain);
            ulong valid_mask = (valid_bits == 64u) ? ~0ul : ((1ul << valid_bits) - 1ul);
            ulong word = bits[row_off + w] & valid_mask;
            uint hit = popcount(word);
            local_n += hit;
            if (hit == 0u) {
                continue;
            }
            if ((hit << 1) > valid_bits) {
                ulong miss = (~word) & valid_mask;
                float miss_sum = 0.0f;
                while (miss != 0ul) {
                    uint tz = ctz(miss);
                    miss_sum += y[sample_base + tz];
                    miss &= (miss - 1ul);
                }
                local_sum += y_word_sums[w] - miss_sum;
            } else {
                ulong hit_word = word;
                while (hit_word != 0ul) {
                    uint tz = ctz(hit_word);
                    local_sum += y[sample_base + tz];
                    hit_word &= (hit_word - 1ul);
                }
            }
        }

        partial_sum[lid] = local_sum;
        partial_n[lid] = local_n;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint active = lsize; active > 1u; active = (active + 1u) >> 1u) {
            uint step = active >> 1u;
            if (lid < step) {
                partial_sum[lid] += partial_sum[lid + step];
                partial_n[lid] += partial_n[lid + step];
            } else if ((active & 1u) != 0u && lid == 0u) {
                partial_sum[0] += partial_sum[active - 1u];
                partial_n[0] += partial_n[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0u) {
            sum_hit_out[group_id] = partial_sum[0];
            n_hit_out[group_id] = partial_n[0];
        }
    }
    "#;

    const MAX_SCORE_THREADS_PER_THREADGROUP: usize = 256;

    #[inline]
    fn floor_power_of_two(v: usize) -> usize {
        if v <= 1 {
            return 1;
        }
        1usize << ((usize::BITS - 1) - v.leading_zeros())
    }

    pub(crate) struct MetalGarfieldScoreBatch {
        device: Device,
        queue: CommandQueue,
        pipeline_legacy: ComputePipelineState,
        pipeline_opt: ComputePipelineState,
        device_name: String,
        requested_threads_per_threadgroup: Option<usize>,
        y_buf: Option<Buffer>,
        y_capacity_samples: usize,
        bits_buf: Option<Buffer>,
        bits_capacity_words: usize,
        y_word_sum_buf: Option<Buffer>,
        y_word_sum_capacity_words: usize,
        sum_buf: Option<Buffer>,
        n_hit_buf: Option<Buffer>,
        out_capacity_rows: usize,
        cached_y_ptr: usize,
        cached_y_len: usize,
        cached_bits_ptr: usize,
        cached_bits_len: usize,
        cached_y_word_ptr: usize,
        cached_y_word_len: usize,
    }

    impl MetalGarfieldScoreBatch {
        pub(crate) fn new(
            requested_threads_per_threadgroup: Option<usize>,
        ) -> Result<Self, String> {
            let device = Device::system_default()
                .or_else(|| Device::all().into_iter().next())
                .ok_or_else(|| "Metal device unavailable on this macOS host".to_string())?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(KERNEL_SRC, &options)
                .map_err(|e| format!("Metal shader compile failed: {e}"))?;
            let fun_legacy = library
                .get_function("garfield_score_sum_hit_n_hit_batch_legacy", None)
                .map_err(|e| format!("Metal kernel lookup failed: {e}"))?;
            let pipeline_legacy = device
                .new_compute_pipeline_state_with_function(&fun_legacy)
                .map_err(|e| format!("Metal legacy pipeline creation failed: {e}"))?;
            let fun_opt = library
                .get_function("garfield_score_sum_hit_n_hit_batch_opt", None)
                .map_err(|e| format!("Metal optimized kernel lookup failed: {e}"))?;
            let pipeline_opt = device
                .new_compute_pipeline_state_with_function(&fun_opt)
                .map_err(|e| format!("Metal optimized pipeline creation failed: {e}"))?;
            let queue = device.new_command_queue();
            let device_name = device.name().to_string();
            Ok(Self {
                device,
                queue,
                pipeline_legacy,
                pipeline_opt,
                device_name,
                requested_threads_per_threadgroup: requested_threads_per_threadgroup
                    .map(|v| v.max(1)),
                y_buf: None,
                y_capacity_samples: 0,
                bits_buf: None,
                bits_capacity_words: 0,
                y_word_sum_buf: None,
                y_word_sum_capacity_words: 0,
                sum_buf: None,
                n_hit_buf: None,
                out_capacity_rows: 0,
                cached_y_ptr: 0,
                cached_y_len: 0,
                cached_bits_ptr: 0,
                cached_bits_len: 0,
                cached_y_word_ptr: 0,
                cached_y_word_len: 0,
            })
        }

        pub(crate) fn device_name(&self) -> &str {
            self.device_name.as_str()
        }

        fn choose_legacy_threadgroup(&self) -> (usize, usize) {
            let width = self.pipeline_legacy.thread_execution_width() as usize;
            let max_total = self.pipeline_legacy.max_total_threads_per_threadgroup() as usize;
            let tg = min(
                self.requested_threads_per_threadgroup
                    .unwrap_or(MAX_SCORE_THREADS_PER_THREADGROUP),
                min(max_total.max(1), MAX_SCORE_THREADS_PER_THREADGROUP),
            )
            .max(1);
            (tg, width.max(1))
        }

        fn choose_opt_threadgroup(&self, row_words: usize) -> (usize, usize) {
            let width = self.pipeline_opt.thread_execution_width() as usize;
            let max_total = self.pipeline_opt.max_total_threads_per_threadgroup() as usize;
            let upper = min(max_total.max(1), MAX_SCORE_THREADS_PER_THREADGROUP).max(1);
            let requested = if let Some(v) = self.requested_threads_per_threadgroup {
                min(v.max(1), upper)
            } else {
                let mul = if row_words >= 128 {
                    8usize
                } else if row_words >= 64 {
                    4usize
                } else if row_words >= 32 {
                    2usize
                } else {
                    1usize
                };
                min(width.max(1).saturating_mul(mul), upper)
            };
            let mut tg = floor_power_of_two(requested.max(1));
            if width > 1 && tg < width {
                tg = width.min(upper);
            }
            tg = tg.min(upper).max(1);
            (tg, width.max(1))
        }

        fn ensure_y_buffer(&mut self, n_samples: usize) {
            if self.y_capacity_samples >= n_samples {
                return;
            }
            self.y_buf = Some(self.device.new_buffer(
                (n_samples.saturating_mul(size_of::<f32>())) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.y_capacity_samples = n_samples;
            self.cached_y_ptr = 0;
            self.cached_y_len = 0;
        }

        fn ensure_bits_buffer(&mut self, n_words: usize) {
            if self.bits_capacity_words >= n_words {
                return;
            }
            self.bits_buf = Some(self.device.new_buffer(
                (n_words.saturating_mul(size_of::<u64>())) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.bits_capacity_words = n_words;
            self.cached_bits_ptr = 0;
            self.cached_bits_len = 0;
        }

        fn ensure_y_word_sum_buffer(&mut self, n_words: usize) {
            if self.y_word_sum_capacity_words >= n_words {
                return;
            }
            self.y_word_sum_buf = Some(self.device.new_buffer(
                (n_words.saturating_mul(size_of::<f32>())) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.y_word_sum_capacity_words = n_words;
            self.cached_y_word_ptr = 0;
            self.cached_y_word_len = 0;
        }

        fn ensure_output_buffers(&mut self, n_rows: usize) {
            if self.out_capacity_rows >= n_rows {
                return;
            }
            self.sum_buf = Some(self.device.new_buffer(
                (n_rows.saturating_mul(size_of::<f32>())) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.n_hit_buf = Some(self.device.new_buffer(
                (n_rows.saturating_mul(size_of::<u32>())) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            self.out_capacity_rows = n_rows;
        }

        fn upload_y_if_needed(&mut self, y: &[f32]) {
            self.ensure_y_buffer(y.len());
            let ptr_key = y.as_ptr() as usize;
            if self.cached_y_ptr == ptr_key && self.cached_y_len == y.len() {
                return;
            }
            let y_buf = self.y_buf.as_ref().expect("y buffer allocated");
            unsafe {
                ptr::copy_nonoverlapping(y.as_ptr(), y_buf.contents() as *mut f32, y.len());
            }
            self.cached_y_ptr = ptr_key;
            self.cached_y_len = y.len();
        }

        fn upload_bits_if_needed(&mut self, bits_flat: &[u64]) {
            self.ensure_bits_buffer(bits_flat.len());
            let ptr_key = bits_flat.as_ptr() as usize;
            if self.cached_bits_ptr == ptr_key && self.cached_bits_len == bits_flat.len() {
                return;
            }
            let bits_buf = self.bits_buf.as_ref().expect("bits buffer allocated");
            unsafe {
                ptr::copy_nonoverlapping(
                    bits_flat.as_ptr(),
                    bits_buf.contents() as *mut u64,
                    bits_flat.len(),
                );
            }
            self.cached_bits_ptr = ptr_key;
            self.cached_bits_len = bits_flat.len();
        }

        fn upload_y_word_sums_if_needed(&mut self, y_word_sums: &[f32]) {
            self.ensure_y_word_sum_buffer(y_word_sums.len());
            let ptr_key = y_word_sums.as_ptr() as usize;
            if self.cached_y_word_ptr == ptr_key && self.cached_y_word_len == y_word_sums.len() {
                return;
            }
            let y_word_sum_buf = self
                .y_word_sum_buf
                .as_ref()
                .expect("y-word-sum buffer allocated");
            unsafe {
                ptr::copy_nonoverlapping(
                    y_word_sums.as_ptr(),
                    y_word_sum_buf.contents() as *mut f32,
                    y_word_sums.len(),
                );
            }
            self.cached_y_word_ptr = ptr_key;
            self.cached_y_word_len = y_word_sums.len();
        }

        pub(crate) fn run_sum_hit_n_hit_legacy(
            &self,
            y: &[f32],
            bits_flat: &[u64],
            row_words: usize,
            n_rows: usize,
            n_samples: usize,
        ) -> Result<(Vec<f64>, Vec<u32>, GpuRunMeta), String> {
            if n_rows == 0 {
                let (_tg, width) = self.choose_legacy_threadgroup();
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    GpuRunMeta {
                        variant: "legacy",
                        y_precision: GPU_Y_PRECISION_NOTE,
                        actual_threads_per_threadgroup: 1,
                        thread_execution_width: width,
                    },
                ));
            }
            let y_buf = self.device.new_buffer(
                (y.len() * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let bits_buf = self.device.new_buffer(
                (bits_flat.len() * size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let sum_buf = self.device.new_buffer(
                (n_rows * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let n_hit_buf = self.device.new_buffer(
                (n_rows * size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                ptr::copy_nonoverlapping(y.as_ptr(), y_buf.contents() as *mut f32, y.len());
                ptr::copy_nonoverlapping(
                    bits_flat.as_ptr(),
                    bits_buf.contents() as *mut u64,
                    bits_flat.len(),
                );
            }
            let prm = ScoreParams {
                n_rows: n_rows as u32,
                row_words: row_words as u32,
                n_samples: n_samples as u32,
                _pad: 0,
            };

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_legacy);
            enc.set_buffer(0, Some(&y_buf), 0);
            enc.set_buffer(1, Some(&bits_buf), 0);
            enc.set_buffer(2, Some(&sum_buf), 0);
            enc.set_buffer(3, Some(&n_hit_buf), 0);
            enc.set_bytes(
                4,
                size_of::<ScoreParams>() as u64,
                &prm as *const ScoreParams as *const c_void,
            );

            let (tg, width) = self.choose_legacy_threadgroup();
            enc.dispatch_threads(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg as u64, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let mut sum_hit_f32 = vec![0.0f32; n_rows];
            let mut n_hit = vec![0u32; n_rows];
            unsafe {
                ptr::copy_nonoverlapping(
                    sum_buf.contents() as *const f32,
                    sum_hit_f32.as_mut_ptr(),
                    n_rows,
                );
                ptr::copy_nonoverlapping(
                    n_hit_buf.contents() as *const u32,
                    n_hit.as_mut_ptr(),
                    n_rows,
                );
            }
            let sum_hit = sum_hit_f32
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            Ok((
                sum_hit,
                n_hit,
                GpuRunMeta {
                    variant: "legacy",
                    y_precision: GPU_Y_PRECISION_NOTE,
                    actual_threads_per_threadgroup: tg,
                    thread_execution_width: width,
                },
            ))
        }

        pub(crate) fn run_sum_hit_n_hit_optimized(
            &mut self,
            y: &[f32],
            bits_flat: &[u64],
            y_word_sums: &[f32],
            row_words: usize,
            n_rows: usize,
            n_samples: usize,
        ) -> Result<(Vec<f64>, Vec<u32>, GpuRunMeta), String> {
            if n_rows == 0 {
                let (_tg, width) = self.choose_opt_threadgroup(row_words);
                return Ok((
                    Vec::new(),
                    Vec::new(),
                    GpuRunMeta {
                        variant: "optimized",
                        y_precision: GPU_Y_PRECISION_NOTE,
                        actual_threads_per_threadgroup: 1,
                        thread_execution_width: width,
                    },
                ));
            }

            self.upload_y_if_needed(y);
            self.upload_bits_if_needed(bits_flat);
            self.upload_y_word_sums_if_needed(y_word_sums);
            self.ensure_output_buffers(n_rows);

            let prm = ScoreParams {
                n_rows: n_rows as u32,
                row_words: row_words as u32,
                n_samples: n_samples as u32,
                _pad: 0,
            };

            let (tg, width) = self.choose_opt_threadgroup(row_words);
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_opt);
            enc.set_buffer(0, self.y_buf.as_ref().map(|v| &**v), 0);
            enc.set_buffer(1, self.bits_buf.as_ref().map(|v| &**v), 0);
            enc.set_buffer(2, self.y_word_sum_buf.as_ref().map(|v| &**v), 0);
            enc.set_buffer(3, self.sum_buf.as_ref().map(|v| &**v), 0);
            enc.set_buffer(4, self.n_hit_buf.as_ref().map(|v| &**v), 0);
            enc.set_bytes(
                5,
                size_of::<ScoreParams>() as u64,
                &prm as *const ScoreParams as *const c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg as u64, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let mut sum_hit_f32 = vec![0.0f32; n_rows];
            let mut n_hit = vec![0u32; n_rows];
            let sum_buf = self.sum_buf.as_ref().expect("sum buffer allocated");
            let n_hit_buf = self.n_hit_buf.as_ref().expect("n_hit buffer allocated");
            unsafe {
                ptr::copy_nonoverlapping(
                    sum_buf.contents() as *const f32,
                    sum_hit_f32.as_mut_ptr(),
                    n_rows,
                );
                ptr::copy_nonoverlapping(
                    n_hit_buf.contents() as *const u32,
                    n_hit.as_mut_ptr(),
                    n_rows,
                );
            }
            let sum_hit = sum_hit_f32
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            Ok((
                sum_hit,
                n_hit,
                GpuRunMeta {
                    variant: "optimized",
                    y_precision: GPU_Y_PRECISION_NOTE,
                    actual_threads_per_threadgroup: tg,
                    thread_execution_width: width,
                },
            ))
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
mod metal_impl {
    pub(crate) struct MetalGarfieldScoreBatch;

    impl MetalGarfieldScoreBatch {
        pub(crate) fn new(_threads_per_threadgroup: Option<usize>) -> Result<Self, String> {
            Err(
                "garfield Metal score batch is unavailable; the installed JanusX was built without \
metal-gpu. Reinstall on macOS from this source tree with `pip install .` \
(the PEP517 backend now auto-enables Metal) or run `maturin develop --release --features metal-gpu`."
                    .to_string(),
            )
        }
    }
}

fn score_cont_centered_gain_batch_packed_metal_impl(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    threads_per_threadgroup: Option<usize>,
) -> Result<(BatchScoreParts, String, GpuRunMeta), String> {
    const CTX: &str = "garfield_score_cont_centered_gain_batch_packed_metal";
    validate_batch_shape(y, row_words, n_rows, n_samples, CTX)?;
    if bits_flat.len() != row_words.saturating_mul(n_rows) {
        return Err(format!(
            "{CTX}: bits size mismatch: got {}, expected {} (n_rows={} row_words={})",
            bits_flat.len(),
            row_words.saturating_mul(n_rows),
            n_rows,
            row_words
        ));
    }
    let y_f32 = y
        .iter()
        .take(n_samples)
        .map(|v| *v as f32)
        .collect::<Vec<_>>();
    let y_word_sums = build_y_word_sums_f32(y_f32.as_slice(), row_words, n_samples);
    let total_sum = y_f32.iter().map(|v| *v as f64).sum::<f64>();
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let scanner = metal_impl::MetalGarfieldScoreBatch::new(threads_per_threadgroup)?;
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _scanner = metal_impl::MetalGarfieldScoreBatch::new(threads_per_threadgroup)?;
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let device_name = scanner.device_name().to_string();
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let device_name = "unavailable".to_string();
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let (sum_hit, n_hit, meta) = {
        let mut scanner = scanner;
        scanner.run_sum_hit_n_hit_optimized(
            y_f32.as_slice(),
            bits_flat,
            y_word_sums.as_slice(),
            row_words,
            n_rows,
            n_samples,
        )?
    };
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let (sum_hit, n_hit, meta) = (
        Vec::new(),
        Vec::new(),
        GpuRunMeta {
            variant: "optimized",
            y_precision: GPU_Y_PRECISION_NOTE,
            actual_threads_per_threadgroup: 0,
            thread_execution_width: 0,
        },
    );
    let parts = build_score_parts_from_sum_hit(sum_hit, n_hit, n_samples, total_sum, row_words);
    Ok((parts, device_name, meta))
}

#[pyfunction(name = "garfield_score_cont_centered_gain_batch_packed_metal")]
pub fn garfield_score_cont_centered_gain_batch_packed_metal_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: usize,
    threads_per_threadgroup: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let yv = y.as_slice()?;
    let shape = bits.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("bits must be a 2D uint64 array"));
    }
    let n_rows = shape[0];
    let row_words = shape[1];
    let bits_flat = bits
        .as_slice()
        .map_err(|_| PyValueError::new_err("bits must be contiguous (C-order)"))?;
    let t0 = Instant::now();
    let (parts, device_name, meta) = py
        .detach(|| {
            score_cont_centered_gain_batch_packed_metal_impl(
                yv,
                bits_flat,
                row_words,
                n_rows,
                n_samples,
                threads_per_threadgroup,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let out = batch_score_parts_to_pydict(py, parts, "metal", Some(elapsed_ms))?;
    out.set_item("device_name", device_name)?;
    out.set_item("variant", meta.variant)?;
    out.set_item("threads_per_threadgroup_requested", threads_per_threadgroup)?;
    out.set_item(
        "threads_per_threadgroup_actual",
        meta.actual_threads_per_threadgroup,
    )?;
    out.set_item("thread_execution_width", meta.thread_execution_width)?;
    out.set_item("y_precision", meta.y_precision)?;
    Ok(out)
}

#[pyfunction(name = "garfield_compare_score_cont_centered_gain_batch_metal_vs_cpu")]
pub fn garfield_compare_score_cont_centered_gain_batch_metal_vs_cpu_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: usize,
    repeats: Option<usize>,
    warmup: Option<usize>,
    threads_per_threadgroup: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let yv = y.as_slice()?;
    let shape = bits.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("bits must be a 2D uint64 array"));
    }
    let n_rows = shape[0];
    let row_words = shape[1];
    let bits_flat = bits
        .as_slice()
        .map_err(|_| PyValueError::new_err("bits must be contiguous (C-order)"))?;
    let reps = repeats.unwrap_or(3).max(1);
    let warm = warmup.unwrap_or(1);

    let (
        cpu_parts,
        gpu_legacy_parts,
        gpu_opt_parts,
        cpu_ms,
        gpu_legacy_ms,
        gpu_opt_ms,
        device_name,
        legacy_meta,
        opt_meta,
    ) = py
        .detach(|| {
            let y_f32_as_f64 = yv
                .iter()
                .take(n_samples)
                .map(|v| (*v as f32) as f64)
                .collect::<Vec<_>>();
            let y_f32 = y_f32_as_f64.iter().map(|v| *v as f32).collect::<Vec<_>>();
            let y_word_sums = build_y_word_sums_f32(y_f32.as_slice(), row_words, n_samples);
            let mut cpu_last: Option<BatchScoreParts> = None;
            let mut gpu_legacy_last: Option<BatchScoreParts> = None;
            let mut gpu_opt_last: Option<BatchScoreParts> = None;
            let mut cpu_times = Vec::with_capacity(reps);
            let mut gpu_legacy_times = Vec::with_capacity(reps);
            let mut gpu_opt_times = Vec::with_capacity(reps);
            let total_sum = y_f32_as_f64.iter().copied().sum::<f64>();
            let mut scanner = metal_impl::MetalGarfieldScoreBatch::new(threads_per_threadgroup)?;
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            let device_name = scanner.device_name().to_string();
            #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
            let device_name = "unavailable".to_string();
            let mut legacy_meta_last: Option<GpuRunMeta> = None;
            let mut opt_meta_last: Option<GpuRunMeta> = None;

            for _ in 0..warm {
                let _ = score_cont_centered_gain_batch_packed_cpu_impl(
                    y_f32_as_f64.as_slice(),
                    bits_flat,
                    row_words,
                    n_rows,
                    n_samples,
                )?;
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                {
                    let (sum_hit_legacy, n_hit_legacy, _legacy_meta) = scanner
                        .run_sum_hit_n_hit_legacy(
                            y_f32.as_slice(),
                            bits_flat,
                            row_words,
                            n_rows,
                            n_samples,
                        )?;
                    let _ = build_score_parts_from_sum_hit(
                        sum_hit_legacy,
                        n_hit_legacy,
                        n_samples,
                        total_sum,
                        row_words,
                    );
                    let (sum_hit_opt, n_hit_opt, _opt_meta) = scanner.run_sum_hit_n_hit_optimized(
                        y_f32.as_slice(),
                        bits_flat,
                        y_word_sums.as_slice(),
                        row_words,
                        n_rows,
                        n_samples,
                    )?;
                    let _ = build_score_parts_from_sum_hit(
                        sum_hit_opt,
                        n_hit_opt,
                        n_samples,
                        total_sum,
                        row_words,
                    );
                }
            }

            for _ in 0..reps {
                let t0 = Instant::now();
                let cpu = score_cont_centered_gain_batch_packed_cpu_impl(
                    y_f32_as_f64.as_slice(),
                    bits_flat,
                    row_words,
                    n_rows,
                    n_samples,
                )?;
                cpu_times.push(t0.elapsed().as_secs_f64() * 1000.0);
                cpu_last = Some(cpu);

                let t1 = Instant::now();
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                let (sum_hit_legacy, n_hit_legacy, legacy_meta) = scanner
                    .run_sum_hit_n_hit_legacy(
                        y_f32.as_slice(),
                        bits_flat,
                        row_words,
                        n_rows,
                        n_samples,
                    )?;
                #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
                let (sum_hit_legacy, n_hit_legacy, legacy_meta) = (
                    Vec::new(),
                    Vec::new(),
                    GpuRunMeta {
                        variant: "legacy",
                        y_precision: GPU_Y_PRECISION_NOTE,
                        actual_threads_per_threadgroup: 0,
                        thread_execution_width: 0,
                    },
                );
                gpu_legacy_times.push(t1.elapsed().as_secs_f64() * 1000.0);
                gpu_legacy_last = Some(build_score_parts_from_sum_hit(
                    sum_hit_legacy,
                    n_hit_legacy,
                    n_samples,
                    total_sum,
                    row_words,
                ));
                legacy_meta_last = Some(legacy_meta);

                let t2 = Instant::now();
                #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                let (sum_hit_opt, n_hit_opt, opt_meta) = scanner.run_sum_hit_n_hit_optimized(
                    y_f32.as_slice(),
                    bits_flat,
                    y_word_sums.as_slice(),
                    row_words,
                    n_rows,
                    n_samples,
                )?;
                #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
                let (sum_hit_opt, n_hit_opt, opt_meta) = (
                    Vec::new(),
                    Vec::new(),
                    GpuRunMeta {
                        variant: "optimized",
                        y_precision: GPU_Y_PRECISION_NOTE,
                        actual_threads_per_threadgroup: 0,
                        thread_execution_width: 0,
                    },
                );
                gpu_opt_times.push(t2.elapsed().as_secs_f64() * 1000.0);
                gpu_opt_last = Some(build_score_parts_from_sum_hit(
                    sum_hit_opt,
                    n_hit_opt,
                    n_samples,
                    total_sum,
                    row_words,
                ));
                opt_meta_last = Some(opt_meta);
            }

            let cpu_last =
                cpu_last.ok_or_else(|| "CPU batch benchmark produced no result".to_string())?;
            let gpu_legacy_last = gpu_legacy_last
                .ok_or_else(|| "GPU legacy batch benchmark produced no result".to_string())?;
            let gpu_opt_last = gpu_opt_last
                .ok_or_else(|| "GPU optimized batch benchmark produced no result".to_string())?;
            let legacy_meta_last =
                legacy_meta_last.ok_or_else(|| "GPU legacy run metadata missing".to_string())?;
            let opt_meta_last =
                opt_meta_last.ok_or_else(|| "GPU optimized run metadata missing".to_string())?;
            Ok::<_, String>((
                cpu_last,
                gpu_legacy_last,
                gpu_opt_last,
                cpu_times,
                gpu_legacy_times,
                gpu_opt_times,
                device_name,
                legacy_meta_last,
                opt_meta_last,
            ))
        })
        .map_err(PyRuntimeError::new_err)?;

    let cpu_elapsed_ms = cpu_ms.iter().copied().sum::<f64>() / (cpu_ms.len() as f64);
    let gpu_legacy_elapsed_ms =
        gpu_legacy_ms.iter().copied().sum::<f64>() / (gpu_legacy_ms.len() as f64);
    let gpu_opt_elapsed_ms = gpu_opt_ms.iter().copied().sum::<f64>() / (gpu_opt_ms.len() as f64);
    let n_rows = cpu_parts
        .n_rows
        .min(gpu_legacy_parts.n_rows)
        .min(gpu_opt_parts.n_rows);
    let legacy_diff = diff_batch_parts(&cpu_parts, &gpu_legacy_parts);
    let opt_diff = diff_batch_parts(&cpu_parts, &gpu_opt_parts);
    let old_new_diff = diff_batch_parts(&gpu_legacy_parts, &gpu_opt_parts);

    let out = PyDict::new(py).into_bound();
    out.set_item("backend", "metal")?;
    out.set_item("device_name", device_name)?;
    out.set_item("n_rows", n_rows)?;
    out.set_item("row_words", cpu_parts.row_words)?;
    out.set_item("n_samples", cpu_parts.n_samples)?;
    out.set_item("repeats", reps)?;
    out.set_item("warmup", warm)?;
    out.set_item("threads_per_threadgroup_requested", threads_per_threadgroup)?;
    out.set_item("cpu_elapsed_ms", cpu_elapsed_ms)?;
    out.set_item("gpu_elapsed_ms", gpu_opt_elapsed_ms)?;
    out.set_item("gpu_legacy_elapsed_ms", gpu_legacy_elapsed_ms)?;
    out.set_item("gpu_optimized_elapsed_ms", gpu_opt_elapsed_ms)?;
    out.set_item("speedup", cpu_elapsed_ms / gpu_opt_elapsed_ms.max(1e-12))?;
    out.set_item(
        "gpu_legacy_speedup",
        cpu_elapsed_ms / gpu_legacy_elapsed_ms.max(1e-12),
    )?;
    out.set_item(
        "gpu_optimized_speedup",
        cpu_elapsed_ms / gpu_opt_elapsed_ms.max(1e-12),
    )?;
    out.set_item(
        "gpu_old_to_new_speedup",
        gpu_legacy_elapsed_ms / gpu_opt_elapsed_ms.max(1e-12),
    )?;
    out.set_item("gpu_variant", opt_meta.variant)?;
    out.set_item("gpu_legacy_variant", legacy_meta.variant)?;
    out.set_item("gpu_optimized_variant", opt_meta.variant)?;
    out.set_item(
        "gpu_legacy_threads_per_threadgroup_actual",
        legacy_meta.actual_threads_per_threadgroup,
    )?;
    out.set_item(
        "gpu_optimized_threads_per_threadgroup_actual",
        opt_meta.actual_threads_per_threadgroup,
    )?;
    out.set_item(
        "thread_execution_width",
        opt_meta
            .thread_execution_width
            .max(legacy_meta.thread_execution_width),
    )?;
    out.set_item(
        "gpu_legacy_thread_execution_width",
        legacy_meta.thread_execution_width,
    )?;
    out.set_item(
        "gpu_optimized_thread_execution_width",
        opt_meta.thread_execution_width,
    )?;
    out.set_item("max_abs_sum_hit_diff", opt_diff.max_abs_sum_hit_diff)?;
    out.set_item("mean_abs_sum_hit_diff", opt_diff.mean_abs_sum_hit_diff)?;
    out.set_item("max_abs_raw_score_diff", opt_diff.max_abs_raw_score_diff)?;
    out.set_item("mean_abs_raw_score_diff", opt_diff.mean_abs_raw_score_diff)?;
    out.set_item(
        "max_abs_support_frac_diff",
        opt_diff.max_abs_support_frac_diff,
    )?;
    out.set_item(
        "legacy_max_abs_sum_hit_diff",
        legacy_diff.max_abs_sum_hit_diff,
    )?;
    out.set_item(
        "legacy_mean_abs_sum_hit_diff",
        legacy_diff.mean_abs_sum_hit_diff,
    )?;
    out.set_item(
        "legacy_max_abs_raw_score_diff",
        legacy_diff.max_abs_raw_score_diff,
    )?;
    out.set_item(
        "legacy_mean_abs_raw_score_diff",
        legacy_diff.mean_abs_raw_score_diff,
    )?;
    out.set_item(
        "legacy_max_abs_support_frac_diff",
        legacy_diff.max_abs_support_frac_diff,
    )?;
    out.set_item(
        "old_new_max_abs_sum_hit_diff",
        old_new_diff.max_abs_sum_hit_diff,
    )?;
    out.set_item(
        "old_new_mean_abs_sum_hit_diff",
        old_new_diff.mean_abs_sum_hit_diff,
    )?;
    out.set_item(
        "old_new_max_abs_raw_score_diff",
        old_new_diff.max_abs_raw_score_diff,
    )?;
    out.set_item(
        "old_new_mean_abs_raw_score_diff",
        old_new_diff.mean_abs_raw_score_diff,
    )?;
    out.set_item(
        "old_new_max_abs_support_frac_diff",
        old_new_diff.max_abs_support_frac_diff,
    )?;
    out.set_item("cpu_elapsed_ms_repeats", cpu_ms)?;
    out.set_item("gpu_elapsed_ms_repeats", gpu_opt_ms.clone())?;
    out.set_item("gpu_legacy_elapsed_ms_repeats", gpu_legacy_ms)?;
    out.set_item("gpu_optimized_elapsed_ms_repeats", gpu_opt_ms)?;
    out.set_item("y_precision", opt_meta.y_precision)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_cpu_matches_scalar() {
        let y = vec![5.0, 4.0, -1.0, -2.0];
        let bits = vec![0b0011_u64, 0b0101_u64];
        let got = score_cont_centered_gain_batch_packed_cpu_impl(&y, &bits, 1, 2, y.len())
            .expect("cpu batch");
        assert_eq!(got.n_hit, vec![2, 2]);
        assert!((got.mean_hit[0] - 4.5).abs() < 1e-12);
        assert!((got.raw_score[0] - 36.0).abs() < 1e-12);
        let sc2 = score_cont_centered_gain_from_sum_and_n_hit(6.0, 4.0, 4, 2);
        assert!((got.raw_score[1] - sc2.raw_score).abs() < 1e-12);
    }
}
