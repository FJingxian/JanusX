use anyhow::Result;
use std::sync::Arc;

pub type ProgressFn = Arc<dyn Fn(u64, u64) -> Result<()> + Send + Sync + 'static>;
