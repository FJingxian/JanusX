//! Generic double-buffer pipeline for overlapping producer/consumer work.
//!
//! Pattern: a background thread produces filled buffers while the calling thread
//! consumes them. This overlaps decode/IO (producer) with BLAS/compute (consumer).
//!
//! # Example
//!
//! ```ignore
//! use crate::io::pipeline::run_double_buffer;
//!
//! struct Chunk { data: Vec<f32>, rows: usize }
//!
//! run_double_buffer(
//!     2,                                          // 2 buffers
//!     || Chunk { data: vec![0.0; 1024], rows: 0 },
//!     |buf| {                                     // producer (bg thread)
//!         // fill buf.data, set buf.rows
//!         true                                    // true = more to come
//!     },
//!     |buf| {                                     // consumer (calling thread)
//!         // process buf.data[..buf.rows]
//!         Ok::<(), String>(())
//!     },
//! )?;
//! ```

use std::sync::mpsc;
use std::thread;

/// Run a double-buffer pipeline with `depth` pre-allocated buffers.
///
/// The producer runs in a background `thread::scope` thread. The consumer runs
/// in the calling thread. When the producer returns `false`, no more buffers are
/// queued; the consumer drains remaining buffers and the pipeline returns.
///
/// If the consumer returns `Err`, the producer is cancelled (channels are dropped)
/// and the error propagates.
///
/// # Arguments
///
/// * `depth` - Number of buffers (2 = double-buffer). Must be >= 2.
/// * `make_buffer` - Factory called `depth` times to seed the free channel.
/// * `producer` - Called in background thread. Receives `&mut B`. Fill the buffer,
///   then return `true` to continue or `false` when done.
/// * `consumer` - Called in calling thread. Receives `&mut B`. Process the buffer,
///   return `Ok(())` to continue or `Err(E)` to abort.
pub fn run_double_buffer<B, M, P, C, E>(
    depth: usize,
    make_buffer: M,
    mut producer: P,
    mut consumer: C,
) -> Result<(), E>
where
    B: Send,
    M: Fn() -> B,
    P: FnMut(&mut B) -> bool + Send,
    C: FnMut(&mut B) -> Result<(), E>,
{
    assert!(depth >= 2, "pipeline depth must be at least 2");

    let (free_tx, free_rx) = mpsc::sync_channel::<B>(depth);
    let (ready_tx, ready_rx) = mpsc::sync_channel::<B>(depth);

    // Seed free channel
    for _ in 0..depth {
        free_tx
            .send(make_buffer())
            .expect("pipeline seed free channel");
    }

    thread::scope(|scope| {
        // Background producer
        scope.spawn(move || {
            while let Ok(mut buf) = free_rx.recv() {
                let more = producer(&mut buf);
                if ready_tx.send(buf).is_err() || !more {
                    break;
                }
            }
        });

        // Foreground consumer
        while let Ok(mut buf) = ready_rx.recv() {
            consumer(&mut buf)?;
            let _ = free_tx.send(buf);
        }

        Ok::<(), E>(())
    })?;

    Ok(())
}
