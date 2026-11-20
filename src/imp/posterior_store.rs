//! Lossless, bounded-RAM handoff from sample HMM jobs to marker output tiles.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STORE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct PosteriorStore {
    file: Option<File>,
    path: PathBuf,
    markers: usize,
    haplotypes: usize,
    expected_haplotypes: usize,
    tile_width: usize,
}

impl PosteriorStore {
    pub(crate) fn new(
        markers: usize,
        tile_width: usize,
        expected_haplotypes: usize,
    ) -> Result<Self, String> {
        if markers == 0 || tile_width == 0 || expected_haplotypes == 0 {
            return Err("posterior store requires positive marker count".to_owned());
        }
        let root = std::env::var_os("JANUSX_RUNTIME_CACHE_DIR")
            .or_else(|| std::env::var_os("JANUSX_CACHE_DIR"))
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&root)
            .map_err(|e| format!("create IMP posterior cache {}: {e}", root.display()))?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("name IMP posterior cache: {e}"))?
            .as_nanos();
        for _ in 0..100 {
            let id = NEXT_STORE.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "jx-imp-posterior-{}-{nonce}-{id}.bin",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        file: Some(file),
                        path,
                        markers,
                        haplotypes: 0,
                        expected_haplotypes,
                        tile_width,
                    })
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(format!(
                        "create IMP posterior cache {}: {e}",
                        path.display()
                    ))
                }
            }
        }
        Err("cannot allocate a unique IMP posterior cache".to_owned())
    }

    pub(crate) fn append(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() % self.markers != 0 {
            return Err("posterior cache batch dimensions are invalid".to_owned());
        }
        let haps = values.len() / self.markers;
        let total = self
            .haplotypes
            .checked_add(haps)
            .ok_or_else(|| "posterior haplotype count overflow".to_owned())?;
        if total > self.expected_haplotypes {
            return Err("posterior cache has too many haplotypes".to_owned());
        }
        let mut tile = Vec::<f32>::new();
        for start in (0..self.markers).step_by(self.tile_width) {
            let end = (start + self.tile_width).min(self.markers);
            tile.clear();
            for row in values.chunks_exact(self.markers) {
                tile.extend_from_slice(&row[start..end]);
            }
            let file = self.file.as_mut().unwrap();
            // Reserve the full tile layout up front by address. Each batch
            // writes only its own haplotypes, making the final tile contiguous.
            let offset = start
                .checked_mul(self.expected_haplotypes)
                .and_then(|v| {
                    self.haplotypes
                        .checked_mul(end - start)
                        .and_then(|n| v.checked_add(n))
                })
                .and_then(|v| v.checked_mul(4))
                .and_then(|v| u64::try_from(v).ok())
                .ok_or_else(|| "posterior cache offset overflow".to_owned())?;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| format!("seek IMP posterior cache {}: {e}", self.path.display()))?;
            // SAFETY: initialized native f32 storage is copied losslessly.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    tile.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(tile.as_slice()),
                )
            };
            file.write_all(bytes)
                .map_err(|e| format!("write IMP posterior cache {}: {e}", self.path.display()))?;
        }
        self.haplotypes = total;
        Ok(())
    }

    pub(crate) fn read_tile(&mut self, start: usize, end: usize) -> Result<Vec<f32>, String> {
        if start >= end || end > self.markers {
            return Err("posterior tile is outside its window".to_owned());
        }
        if self.haplotypes != self.expected_haplotypes {
            return Err("posterior cache is incomplete".to_owned());
        }
        let width = end - start;
        let tile_index = start / self.tile_width;
        let tile_start = tile_index * self.tile_width;
        let tile_end = (tile_start + self.tile_width).min(self.markers);
        if end > tile_end {
            return Err("posterior read crosses a storage tile".to_owned());
        }
        let stored_width = tile_end - tile_start;
        let count = self
            .haplotypes
            .checked_mul(stored_width)
            .ok_or_else(|| "posterior tile size overflow".to_owned())?;
        let mut batch = vec![0.0f32; count];
        let offset = tile_start
            .checked_mul(self.expected_haplotypes)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| "posterior cache offset overflow".to_owned())?;
        self.file
            .as_mut()
            .unwrap()
            .seek(SeekFrom::Start(offset))
            .map_err(|e| format!("seek IMP posterior cache {}: {e}", self.path.display()))?;
        // SAFETY: all f32 bit patterns are valid. read_exact fills the
        // initialized buffer only with the native f32 bytes written above.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                batch.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(batch.as_slice()),
            )
        };
        self.file
            .as_mut()
            .unwrap()
            .read_exact(bytes)
            .map_err(|e| format!("read IMP posterior cache {}: {e}", self.path.display()))?;
        if width == stored_width {
            return Ok(batch);
        }
        let mut values = Vec::with_capacity(self.haplotypes * width);
        for row in batch.chunks_exact(stored_width) {
            values.extend_from_slice(&row[start - tile_start..end - tile_start]);
        }
        Ok(values)
    }
}

impl Drop for PosteriorStore {
    fn drop(&mut self) {
        // Windows cannot unlink an open file.
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}
