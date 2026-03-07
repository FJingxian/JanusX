use crate::pipeline::{all_outputs_ready, PipelineStep, StepItem};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Default)]
pub(super) struct WorkSummary {
    pub done_items: usize,
    pub total_items: usize,
    pub done_steps: usize,
    pub total_steps: usize,
}

#[derive(Clone, Debug, Default)]
struct WorkStepState {
    name: String,
    done: usize,
    total: usize,
    items: BTreeMap<String, bool>,
    items_completed_at: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
struct WorkState {
    version: usize,
    signature: String,
    params: JsonValue,
    created_at: String,
    updated_at: String,
    status: String,
    started_at: Option<String>,
    completed_at: Option<String>,
    failed_at: Option<String>,
    error: Option<String>,
    steps: BTreeMap<String, WorkStepState>,
    summary: WorkSummary,
}

#[derive(Clone, Debug)]
struct ItemMeta {
    id: String,
    outputs: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
struct StepMeta {
    id: String,
    name: String,
    items: Vec<ItemMeta>,
}

pub(super) struct WorkStateTracker {
    path: PathBuf,
    state: WorkState,
    steps_meta: Vec<StepMeta>,
}

impl WorkStateTracker {
    pub(super) fn init_or_resume(
        state_path: &Path,
        params_compact: &str,
        input_files: &[PathBuf],
        steps: &[PipelineStep],
    ) -> Result<(Self, bool), String> {
        let params_json = parse_json(params_compact)
            .map_err(|e| format!("Failed to parse run params JSON: {e}"))?;
        let signature = signature_from_params(params_compact);
        let steps_meta = build_steps_meta(steps);
        let input_files = dedup_paths(input_files);

        let mut resumed = false;
        let mut state_opt: Option<WorkState> = None;
        if state_path.exists() {
            let old = read_work_state(state_path).ok().flatten();
            let ref_epoch = old
                .as_ref()
                .and_then(|x| parse_utc_iso_to_epoch(&x.created_at))
                .or_else(|| file_mtime_epoch(state_path))
                .unwrap_or(0);
            validate_inputs_not_newer_than_ref(&input_files, ref_epoch, state_path)?;

            if let Some(mut old_state) = old {
                let old_params = to_json_compact(&old_state.params);
                if old_state.signature == signature || old_params == params_compact {
                    old_state.signature = signature.clone();
                    old_state.params = params_json.clone();
                    state_opt = Some(old_state);
                    resumed = true;
                }
            }
        }

        let mut tracker = Self {
            path: state_path.to_path_buf(),
            state: state_opt.unwrap_or_else(|| WorkState::new(signature, params_json)),
            steps_meta,
        };
        tracker.sync_from_fs()?;
        if tracker.state.summary.total_items > 0
            && tracker.state.summary.done_items >= tracker.state.summary.total_items
        {
            tracker.state.status = "completed".to_string();
        } else if tracker.state.status != "failed" {
            tracker.state.status = "running".to_string();
        }
        tracker.write_state()?;
        Ok((tracker, resumed))
    }

    pub(super) fn summary(&self) -> WorkSummary {
        self.state.summary.clone()
    }

    pub(super) fn mark_running(&mut self) -> Result<(), String> {
        self.sync_from_fs()?;
        self.state.status = "running".to_string();
        if self.state.started_at.is_none() {
            self.state.started_at = Some(now_utc_iso());
        }
        self.write_state()
    }

    pub(super) fn mark_item_done(
        &mut self,
        step: &PipelineStep,
        item: &StepItem,
    ) -> Result<(), String> {
        if !all_outputs_ready(&item.outputs) {
            return Ok(());
        }
        let now = now_utc_iso();
        let expected_total = self.expected_step_total(&step.id);
        let entry = self
            .state
            .steps
            .entry(step.id.clone())
            .or_insert_with(|| WorkStepState {
                name: step.name.clone(),
                ..WorkStepState::default()
            });
        if entry.name.trim().is_empty() {
            entry.name = step.name.clone();
        }
        entry.items.insert(item.id.clone(), true);
        entry
            .items_completed_at
            .entry(item.id.clone())
            .or_insert_with(|| now.clone());

        let done = entry.items.values().filter(|x| **x).count();
        entry.done = done;
        entry.total = expected_total.max(entry.total).max(done);

        if self.state.status != "failed" {
            self.state.status = "running".to_string();
        }
        self.refresh_summary();
        self.state.updated_at = now_utc_iso();
        self.write_state()
    }

    pub(super) fn mark_failed(&mut self, error: &str) -> Result<(), String> {
        self.sync_from_fs()?;
        self.state.status = "failed".to_string();
        self.state.error = Some(error.to_string());
        self.state.failed_at = Some(now_utc_iso());
        self.state.updated_at = now_utc_iso();
        self.write_state()
    }

    pub(super) fn mark_completed(&mut self) -> Result<(), String> {
        self.sync_from_fs()?;
        self.state.status = "completed".to_string();
        self.state.error = None;
        self.state.completed_at = Some(now_utc_iso());
        self.state.updated_at = now_utc_iso();
        self.write_state()
    }

    fn sync_from_fs(&mut self) -> Result<(), String> {
        let now = now_utc_iso();
        let mut done_items = 0usize;
        let mut total_items = 0usize;
        let mut done_steps = 0usize;

        for step in &self.steps_meta {
            let prev = self.state.steps.get(&step.id).cloned().unwrap_or_default();
            let mut entry = WorkStepState {
                name: step.name.clone(),
                done: 0,
                total: step.items.len(),
                items: BTreeMap::new(),
                items_completed_at: BTreeMap::new(),
            };
            for item in &step.items {
                let completed = all_outputs_ready(&item.outputs);
                entry.items.insert(item.id.clone(), completed);
                if completed {
                    entry.done += 1;
                    let ts = prev
                        .items_completed_at
                        .get(&item.id)
                        .cloned()
                        .unwrap_or_else(|| now.clone());
                    entry.items_completed_at.insert(item.id.clone(), ts);
                }
            }
            total_items += entry.total;
            done_items += entry.done;
            if entry.total > 0 && entry.done >= entry.total {
                done_steps += 1;
            }
            self.state.steps.insert(step.id.clone(), entry);
        }
        self.state.summary = WorkSummary {
            done_items,
            total_items,
            done_steps,
            total_steps: self.steps_meta.len(),
        };
        self.state.updated_at = now_utc_iso();
        Ok(())
    }

    fn write_state(&self) -> Result<(), String> {
        let value = self.state.to_json();
        let text = to_json_pretty(&value);
        safe_write_atomic(&self.path, &text)
    }

    fn refresh_summary(&mut self) {
        let mut done_items = 0usize;
        let mut total_items = 0usize;
        let mut done_steps = 0usize;
        let total_steps = self.state.steps.len();
        for entry in self.state.steps.values_mut() {
            let done = entry.items.values().filter(|x| **x).count();
            entry.done = done;
            if entry.total < done {
                entry.total = done;
            }
            done_items += entry.done;
            total_items += entry.total;
            if entry.total > 0 && entry.done >= entry.total {
                done_steps += 1;
            }
        }
        self.state.summary = WorkSummary {
            done_items,
            total_items,
            done_steps,
            total_steps,
        };
    }

    fn expected_step_total(&self, step_id: &str) -> usize {
        for step in &self.steps_meta {
            if step.id == step_id {
                return step.items.len();
            }
        }
        0
    }
}

impl WorkState {
    fn new(signature: String, params: JsonValue) -> Self {
        let now = now_utc_iso();
        Self {
            version: 1,
            signature,
            params,
            created_at: now.clone(),
            updated_at: now,
            status: "initialized".to_string(),
            started_at: None,
            completed_at: None,
            failed_at: None,
            error: None,
            steps: BTreeMap::new(),
            summary: WorkSummary::default(),
        }
    }

    fn to_json(&self) -> JsonValue {
        let mut root = BTreeMap::new();
        root.insert(
            "version".to_string(),
            JsonValue::Number(self.version as f64),
        );
        root.insert(
            "signature".to_string(),
            JsonValue::String(self.signature.clone()),
        );
        root.insert("params".to_string(), self.params.clone());
        root.insert(
            "created_at".to_string(),
            JsonValue::String(self.created_at.clone()),
        );
        root.insert(
            "updated_at".to_string(),
            JsonValue::String(self.updated_at.clone()),
        );
        root.insert("status".to_string(), JsonValue::String(self.status.clone()));
        if let Some(v) = &self.started_at {
            root.insert("started_at".to_string(), JsonValue::String(v.clone()));
        }
        if let Some(v) = &self.completed_at {
            root.insert("completed_at".to_string(), JsonValue::String(v.clone()));
        }
        if let Some(v) = &self.failed_at {
            root.insert("failed_at".to_string(), JsonValue::String(v.clone()));
        }
        if let Some(v) = &self.error {
            root.insert("error".to_string(), JsonValue::String(v.clone()));
        }

        let mut steps_obj = BTreeMap::new();
        for (sid, entry) in &self.steps {
            let mut eo = BTreeMap::new();
            eo.insert("name".to_string(), JsonValue::String(entry.name.clone()));
            eo.insert("done".to_string(), JsonValue::Number(entry.done as f64));
            eo.insert("total".to_string(), JsonValue::Number(entry.total as f64));
            let mut items_obj = BTreeMap::new();
            for (iid, done) in &entry.items {
                items_obj.insert(iid.clone(), JsonValue::Bool(*done));
            }
            eo.insert("items".to_string(), JsonValue::Object(items_obj));
            let mut done_at_obj = BTreeMap::new();
            for (iid, ts) in &entry.items_completed_at {
                done_at_obj.insert(iid.clone(), JsonValue::String(ts.clone()));
            }
            eo.insert(
                "items_completed_at".to_string(),
                JsonValue::Object(done_at_obj),
            );
            steps_obj.insert(sid.clone(), JsonValue::Object(eo));
        }
        root.insert("steps".to_string(), JsonValue::Object(steps_obj));

        let mut summary = BTreeMap::new();
        summary.insert(
            "done_items".to_string(),
            JsonValue::Number(self.summary.done_items as f64),
        );
        summary.insert(
            "total_items".to_string(),
            JsonValue::Number(self.summary.total_items as f64),
        );
        summary.insert(
            "done_steps".to_string(),
            JsonValue::Number(self.summary.done_steps as f64),
        );
        summary.insert(
            "total_steps".to_string(),
            JsonValue::Number(self.summary.total_steps as f64),
        );
        root.insert("summary".to_string(), JsonValue::Object(summary));
        JsonValue::Object(root)
    }
}

fn build_steps_meta(steps: &[PipelineStep]) -> Vec<StepMeta> {
    steps
        .iter()
        .map(|s| StepMeta {
            id: s.id.clone(),
            name: s.name.clone(),
            items: s
                .items
                .iter()
                .map(|i| ItemMeta {
                    id: i.id.clone(),
                    outputs: i.outputs.clone(),
                })
                .collect(),
        })
        .collect()
}

fn dedup_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = BTreeMap::<String, ()>::new();
    for p in paths {
        let key = p.to_string_lossy().to_string();
        if seen.contains_key(&key) {
            continue;
        }
        seen.insert(key, ());
        out.push(p.clone());
    }
    out
}

fn validate_inputs_not_newer_than_ref(
    input_files: &[PathBuf],
    ref_epoch: u64,
    state_path: &Path,
) -> Result<(), String> {
    let mut newer: Vec<String> = Vec::new();
    for p in input_files {
        if !p.exists() || !p.is_file() {
            continue;
        }
        let mt = match file_mtime_epoch(p) {
            Some(v) => v,
            None => continue,
        };
        if mt > ref_epoch {
            newer.push(p.to_string_lossy().to_string());
        }
    }
    if newer.is_empty() {
        return Ok(());
    }
    newer.sort();
    let ts = epoch_to_utc_iso(ref_epoch);
    Err(format!(
        "Detected input files newer than existing .work.json (created_at={ts}). Please remove {} to restart.\nNewer inputs:\n- {}",
        state_path.display(),
        newer.join("\n- ")
    ))
}

fn signature_from_params(params: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in params.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn safe_write_atomic(path: &Path, payload: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }
    let tmp = PathBuf::from(format!("{}.tmp", path.to_string_lossy()));
    fs::write(&tmp, payload).map_err(|e| format!("Failed to write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| {
        format!(
            "Failed to move {} -> {}: {e}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn read_work_state(path: &Path) -> Result<Option<WorkState>, String> {
    let text = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let root = match parse_json(&text) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let Some(obj) = as_object(&root) else {
        return Ok(None);
    };

    let version = get_usize(obj, "version").unwrap_or(1);
    let signature = get_string(obj, "signature").unwrap_or_default();
    let params = obj
        .get("params")
        .cloned()
        .unwrap_or_else(|| JsonValue::Object(BTreeMap::new()));
    let created_at = get_string(obj, "created_at").unwrap_or_default();
    let updated_at = get_string(obj, "updated_at").unwrap_or_else(now_utc_iso);
    let status = get_string(obj, "status").unwrap_or_else(|| "initialized".to_string());
    let started_at = get_string(obj, "started_at");
    let completed_at = get_string(obj, "completed_at");
    let failed_at = get_string(obj, "failed_at");
    let error = get_string(obj, "error");

    let mut steps: BTreeMap<String, WorkStepState> = BTreeMap::new();
    if let Some(steps_obj) = obj.get("steps").and_then(as_object) {
        for (sid, sv) in steps_obj {
            let Some(so) = as_object(sv) else {
                continue;
            };
            let mut entry = WorkStepState::default();
            entry.name = get_string(so, "name").unwrap_or_else(|| sid.clone());
            entry.done = get_usize(so, "done").unwrap_or(0);
            entry.total = get_usize(so, "total").unwrap_or(0);
            if let Some(items_obj) = so.get("items").and_then(as_object) {
                for (iid, iv) in items_obj {
                    let done = match iv {
                        JsonValue::Bool(v) => *v,
                        _ => false,
                    };
                    entry.items.insert(iid.clone(), done);
                }
            }
            if let Some(done_at_obj) = so.get("items_completed_at").and_then(as_object) {
                for (iid, tv) in done_at_obj {
                    if let JsonValue::String(ts) = tv {
                        entry.items_completed_at.insert(iid.clone(), ts.clone());
                    }
                }
            }
            steps.insert(sid.clone(), entry);
        }
    }

    let mut summary = WorkSummary::default();
    if let Some(sum_obj) = obj.get("summary").and_then(as_object) {
        summary.done_items = get_usize(sum_obj, "done_items").unwrap_or(0);
        summary.total_items = get_usize(sum_obj, "total_items").unwrap_or(0);
        summary.done_steps = get_usize(sum_obj, "done_steps").unwrap_or(0);
        summary.total_steps = get_usize(sum_obj, "total_steps").unwrap_or(0);
    }

    Ok(Some(WorkState {
        version,
        signature,
        params,
        created_at,
        updated_at,
        status,
        started_at,
        completed_at,
        failed_at,
        error,
        steps,
        summary,
    }))
}

fn get_string(obj: &BTreeMap<String, JsonValue>, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(JsonValue::String(v)) => Some(v.clone()),
        _ => None,
    }
}

fn get_usize(obj: &BTreeMap<String, JsonValue>, key: &str) -> Option<usize> {
    match obj.get(key) {
        Some(JsonValue::Number(v)) if *v >= 0.0 => Some(*v as usize),
        _ => None,
    }
}

fn file_mtime_epoch(path: &Path) -> Option<u64> {
    let md = fs::metadata(path).ok()?;
    let mt = md.modified().ok()?;
    mt.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

fn now_utc_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    epoch_to_utc_iso(secs)
}

fn parse_utc_iso_to_epoch(ts: &str) -> Option<u64> {
    if ts.len() != 20 || !ts.ends_with('Z') {
        return None;
    }
    let y = ts.get(0..4)?.parse::<i32>().ok()?;
    let mon = ts.get(5..7)?.parse::<u32>().ok()?;
    let day = ts.get(8..10)?.parse::<u32>().ok()?;
    let h = ts.get(11..13)?.parse::<u32>().ok()?;
    let min = ts.get(14..16)?.parse::<u32>().ok()?;
    let sec = ts.get(17..19)?.parse::<u32>().ok()?;
    if ts.as_bytes().get(4).copied() != Some(b'-')
        || ts.as_bytes().get(7).copied() != Some(b'-')
        || ts.as_bytes().get(10).copied() != Some(b'T')
        || ts.as_bytes().get(13).copied() != Some(b':')
        || ts.as_bytes().get(16).copied() != Some(b':')
    {
        return None;
    }
    if mon == 0 || mon > 12 || day == 0 || day > 31 || h > 23 || min > 59 || sec > 59 {
        return None;
    }
    let days = days_from_civil(y, mon, day);
    if days < 0 {
        return None;
    }
    let day_secs = u64::from(h) * 3600 + u64::from(min) * 60 + u64::from(sec);
    Some((days as u64) * 86400 + day_secs)
}

fn epoch_to_utc_iso(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (y, m, d) = civil_from_days(days);
    let h = rem / 3600;
    let min = (rem % 3600) / 60;
    let s = rem % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y_adj = i64::from(y) - if m <= 2 { 1 } else { 0 };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let m_adj = i64::from(m) + if m > 2 { -3 } else { 9 };
    let doy = (153 * m_adj + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

#[derive(Clone, Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

fn as_object(v: &JsonValue) -> Option<&BTreeMap<String, JsonValue>> {
    match v {
        JsonValue::Object(o) => Some(o),
        _ => None,
    }
}

fn parse_json(text: &str) -> Result<JsonValue, String> {
    let mut p = JsonParser::new(text);
    let v = p.parse_value()?;
    p.skip_ws();
    if !p.eof() {
        return Err(format!("Unexpected trailing JSON at byte {}", p.pos));
    }
    Ok(v)
}

fn to_json_compact(v: &JsonValue) -> String {
    let mut out = String::new();
    write_json_compact(v, &mut out);
    out
}

fn to_json_pretty(v: &JsonValue) -> String {
    let mut out = String::new();
    write_json_pretty(v, &mut out, 0);
    out.push('\n');
    out
}

fn write_json_compact(v: &JsonValue, out: &mut String) {
    match v {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(n) => out.push_str(&format_json_number(*n)),
        JsonValue::String(s) => {
            out.push('"');
            out.push_str(&json_escape(s));
            out.push('"');
        }
        JsonValue::Array(arr) => {
            out.push('[');
            for (idx, x) in arr.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json_compact(x, out);
            }
            out.push(']');
        }
        JsonValue::Object(obj) => {
            out.push('{');
            let mut first = true;
            for (k, val) in obj {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push('"');
                out.push_str(&json_escape(k));
                out.push_str("\":");
                write_json_compact(val, out);
            }
            out.push('}');
        }
    }
}

fn write_json_pretty(v: &JsonValue, out: &mut String, indent: usize) {
    match v {
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
            write_json_compact(v, out);
        }
        JsonValue::Array(arr) => {
            if arr.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            out.push('\n');
            for (idx, x) in arr.iter().enumerate() {
                out.push_str(&" ".repeat(indent + 2));
                write_json_pretty(x, out, indent + 2);
                if idx + 1 < arr.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&" ".repeat(indent));
            out.push(']');
        }
        JsonValue::Object(obj) => {
            if obj.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            out.push('\n');
            let len = obj.len();
            for (idx, (k, val)) in obj.iter().enumerate() {
                out.push_str(&" ".repeat(indent + 2));
                out.push('"');
                out.push_str(&json_escape(k));
                out.push_str("\": ");
                write_json_pretty(val, out, indent + 2);
                if idx + 1 < len {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&" ".repeat(indent));
            out.push('}');
        }
    }
}

fn format_json_number(v: f64) -> String {
    if (v.fract().abs() < f64::EPSILON) && v.is_finite() {
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\n' | b'\t' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number().map(JsonValue::Number),
            Some(other) => Err(format!(
                "Unexpected byte '{}' at {}",
                other as char, self.pos
            )),
            None => Err("Unexpected end of JSON".to_string()),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], out: JsonValue) -> Result<JsonValue, String> {
        if self.bytes.len() < self.pos + lit.len() {
            return Err("Unexpected end of JSON".to_string());
        }
        if &self.bytes[self.pos..self.pos + lit.len()] != lit {
            return Err(format!("Invalid literal at byte {}", self.pos));
        }
        self.pos += lit.len();
        Ok(out)
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        let mut obj = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(obj));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.parse_value()?;
            obj.insert(key, val);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(format!("Expected ',' or '}}' at byte {}", self.pos)),
            }
        }
        Ok(JsonValue::Object(obj))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(format!("Expected ',' or ']' at byte {}", self.pos)),
            }
        }
        Ok(JsonValue::Array(arr))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let Some(b) = self.next() else {
                return Err("Unexpected end while parsing string".to_string());
            };
            match b {
                b'"' => break,
                b'\\' => {
                    let esc = self
                        .next()
                        .ok_or_else(|| "Unexpected end after escape".to_string())?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0c'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                let saved = self.pos;
                                if self.next() == Some(b'\\') && self.next() == Some(b'u') {
                                    let low = self.parse_hex4()?;
                                    if (0xDC00..=0xDFFF).contains(&low) {
                                        let high_ten = cp - 0xD800;
                                        let low_ten = low - 0xDC00;
                                        let combined = 0x10000 + ((high_ten << 10) | low_ten);
                                        if let Some(ch) = char::from_u32(combined) {
                                            out.push(ch);
                                        }
                                    } else if let Some(ch) = char::from_u32(cp) {
                                        out.push(ch);
                                        self.pos = saved;
                                    }
                                } else if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                    self.pos = saved;
                                }
                            } else if let Some(ch) = char::from_u32(cp) {
                                out.push(ch);
                            }
                        }
                        _ => return Err(format!("Invalid escape at byte {}", self.pos)),
                    }
                }
                c if c < 0x20 => {
                    return Err(format!("Control character in string at {}", self.pos))
                }
                other => {
                    if other < 0x80 {
                        out.push(other as char);
                    } else {
                        let start = self.pos - 1;
                        let width = utf8_width(other);
                        if width == 0 {
                            return Err(format!("Invalid UTF-8 start byte at {}", start));
                        }
                        if self.pos - 1 + width > self.bytes.len() {
                            return Err("Unexpected end in UTF-8 sequence".to_string());
                        }
                        let slice = &self.bytes[start..start + width];
                        let s = std::str::from_utf8(slice)
                            .map_err(|_| format!("Invalid UTF-8 sequence at {start}"))?;
                        out.push_str(s);
                        self.pos = start + width;
                    }
                }
            }
        }
        Ok(out)
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let b = self
                .next()
                .ok_or_else(|| "Unexpected end while parsing \\u escape".to_string())?;
            let d = match b {
                b'0'..=b'9' => u32::from(b - b'0'),
                b'a'..=b'f' => u32::from(b - b'a') + 10,
                b'A'..=b'F' => u32::from(b - b'A') + 10,
                _ => return Err(format!("Invalid hex digit in \\u escape at {}", self.pos)),
            };
            v = (v << 4) | d;
        }
        Ok(v)
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("Invalid number at byte {}", start)),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("Invalid fraction at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("Invalid exponent at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| format!("Invalid number bytes at {start}"))?;
        s.parse::<f64>()
            .map_err(|_| format!("Invalid number literal: {s}"))
    }

    fn expect(&mut self, want: u8) -> Result<(), String> {
        match self.next() {
            Some(got) if got == want => Ok(()),
            _ => Err(format!("Expected '{}' at byte {}", want as char, self.pos)),
        }
    }
}

fn utf8_width(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => 0,
    }
}
