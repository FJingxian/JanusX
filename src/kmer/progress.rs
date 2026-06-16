use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DRAW_INTERVAL: Duration = Duration::from_millis(250);
const BAR_WIDTH: usize = 28;

#[derive(Clone)]
pub struct KmergeProgressBar {
    inner: Arc<Mutex<ProgressState>>,
}

struct ProgressState {
    desc: String,
    total: u64,
    done: u64,
    enabled: bool,
    start: Instant,
    last_draw: Instant,
    last_len: usize,
    finished: bool,
}

impl KmergeProgressBar {
    pub fn new(desc: impl Into<String>, total: u64) -> Self {
        let now = Instant::now();
        Self {
            inner: Arc::new(Mutex::new(ProgressState {
                desc: desc.into(),
                total: total.max(1),
                done: 0,
                enabled: io::stderr().is_terminal(),
                start: now,
                last_draw: now.checked_sub(DRAW_INTERVAL).unwrap_or(now),
                last_len: 0,
                finished: false,
            })),
        }
    }

    pub fn inc(&self, delta: u64) {
        if delta == 0 {
            return;
        }
        let mut state = match self.inner.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        if state.finished {
            return;
        }
        state.done = state.done.saturating_add(delta).min(state.total);
        draw_progress(&mut state, false);
    }

    pub fn finish(&self) {
        let mut state = match self.inner.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        if state.finished {
            return;
        }
        state.done = state.total;
        draw_progress(&mut state, true);
        state.finished = true;
    }
}

fn draw_progress(state: &mut ProgressState, force: bool) {
    if !state.enabled {
        return;
    }
    let now = Instant::now();
    if !force && now.duration_since(state.last_draw) < DRAW_INTERVAL {
        return;
    }
    state.last_draw = now;

    let total = state.total.max(1);
    let done = state.done.min(total);
    let frac = done as f64 / total as f64;
    let filled = ((frac * BAR_WIDTH as f64).round() as usize).min(BAR_WIDTH);
    let mut bar = String::with_capacity(BAR_WIDTH);
    for idx in 0..BAR_WIDTH {
        if idx < filled {
            bar.push('=');
        } else if idx == filled && done < total {
            bar.push('>');
        } else {
            bar.push(' ');
        }
    }

    let elapsed = state.start.elapsed();
    let eta = if done > 0 && done < total {
        let per_unit = elapsed.as_secs_f64() / done as f64;
        Duration::from_secs_f64(per_unit * (total - done) as f64)
    } else {
        Duration::ZERO
    };

    let mut line = format!(
        "{} [{}] {:>6.1}% {:>12}/{} [{}<{}]",
        state.desc,
        bar,
        frac * 100.0,
        done,
        total,
        format_duration(elapsed),
        format_duration(eta),
    );

    if line.len() < state.last_len {
        line.push_str(&" ".repeat(state.last_len - line.len()));
    }
    state.last_len = line.len();

    let mut stderr = io::stderr().lock();
    let _ = write!(stderr, "\r{line}");
    if force {
        let _ = writeln!(stderr);
    }
    let _ = stderr.flush();
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let mins = secs / 60;
    let rem = secs % 60;
    if mins == 0 {
        format!("{rem:02}s")
    } else if mins < 60 {
        format!("{mins:02}m{rem:02}s")
    } else {
        let hours = mins / 60;
        let rem_mins = mins % 60;
        format!("{hours:02}h{rem_mins:02}m")
    }
}
