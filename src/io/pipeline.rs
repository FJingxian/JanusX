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

enum StageMessage<T> {
    Item(T),
    Done,
    Error(String),
}

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

    thread::scope(move |scope| {
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
            if let Err(error) = consumer(&mut buf) {
                // Close both channels before the scope waits for the producer.
                // It may be blocked sending a full ready queue or waiting for a
                // buffer that can no longer be returned by this consumer.
                drop(free_tx);
                drop(ready_rx);
                return Err(error);
            }
            let _ = free_tx.send(buf);
        }

        Ok::<(), E>(())
    })?;

    Ok(())
}

/// Run a bounded three-stage pipeline with recycled consumer buffers.
///
/// Stage 1 produces metadata descriptors on its own thread. Stage 2 consumes
/// those descriptors and fills a recycled work buffer on a second thread. The
/// calling thread consumes completed buffers. This keeps metadata scans,
/// genotype reads/decodes, and compute work independently schedulable while
/// retaining a fixed memory bound.
pub fn run_three_stage<A, B, M, P1, P2, C>(
    depth: usize,
    make_buffer: M,
    mut produce: P1,
    mut decode: P2,
    mut consume: C,
) -> Result<(), String>
where
    A: Send,
    B: Send,
    M: Fn() -> B,
    P1: FnMut() -> Result<Option<A>, String> + Send,
    P2: FnMut(A, &mut B) -> Result<bool, String> + Send,
    C: FnMut(&mut B) -> Result<(), String>,
{
    assert!(depth >= 2, "pipeline depth must be at least 2");

    let (meta_tx, meta_rx) = mpsc::sync_channel::<StageMessage<A>>(2);
    let (free_tx, free_rx) = mpsc::sync_channel::<B>(depth);
    let (ready_tx, ready_rx) = mpsc::sync_channel::<StageMessage<B>>(depth);
    let free_tx_for_decode = free_tx.clone();

    for _ in 0..depth {
        free_tx
            .send(make_buffer())
            .expect("pipeline seed free channel");
    }

    thread::scope(move |scope| {
        scope.spawn(move || loop {
            let message = match produce() {
                Ok(Some(item)) => StageMessage::Item(item),
                Ok(None) => StageMessage::Done,
                Err(error) => StageMessage::Error(error),
            };
            let finished = !matches!(message, StageMessage::Item(_));
            if meta_tx.send(message).is_err() {
                break;
            }
            if finished {
                break;
            }
        });

        scope.spawn(move || {
            while let Ok(message) = meta_rx.recv() {
                match message {
                    StageMessage::Item(item) => {
                        let Ok(mut buffer) = free_rx.recv() else {
                            break;
                        };
                        match decode(item, &mut buffer) {
                            Ok(true) => {
                                if ready_tx.send(StageMessage::Item(buffer)).is_err() {
                                    break;
                                }
                            }
                            Ok(false) => {
                                let _ = free_tx_for_decode.send(buffer);
                                let _ = ready_tx.send(StageMessage::Done);
                                break;
                            }
                            Err(error) => {
                                let _ = ready_tx.send(StageMessage::Error(error));
                                break;
                            }
                        }
                    }
                    StageMessage::Done => {
                        let _ = ready_tx.send(StageMessage::Done);
                        break;
                    }
                    StageMessage::Error(error) => {
                        let _ = ready_tx.send(StageMessage::Error(error));
                        break;
                    }
                }
            }
        });

        while let Ok(message) = ready_rx.recv() {
            match message {
                StageMessage::Item(mut buffer) => {
                    if let Err(error) = consume(&mut buffer) {
                        // Closing both queues releases a producer blocked on a
                        // full metadata queue or a stage-2 thread waiting for
                        // an in-flight consumer buffer.
                        drop(free_tx);
                        drop(ready_rx);
                        return Err(error);
                    }
                    // Stage 2 may have finished and dropped the recycled-buffer
                    // receiver while completed work remains queued. Keep
                    // draining until the terminal message arrives.
                    let _ = free_tx.send(buffer);
                }
                StageMessage::Done => return Ok(()),
                StageMessage::Error(error) => return Err(error),
            }
        }

        Err("three-stage pipeline ended without a completion signal".to_string())
    })
}
