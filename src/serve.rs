use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use crate::pipeline::{Pipeline, PipelineOptions};

enum ExtractError {
    /// Request was syntactically fine but carried unusable data (bad regex,
    /// malformed video_path, …). Maps to 400.
    Client(String),
    /// Upstream fetch (e.g. Frigate) returned 404 — clip not there. Neither
    /// the client's fault nor ours. Maps to 502.
    UpstreamNotFound(String),
    /// Decode / OCR / mutex poisoning / anything server-side. Maps to 500.
    Internal(String),
}

impl ExtractError {
    fn into_response(self) -> (StatusCode, String) {
        match self {
            Self::Client(m) => (StatusCode::BAD_REQUEST, m),
            Self::UpstreamNotFound(m) => (StatusCode::BAD_GATEWAY, m),
            Self::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        }
    }
}

fn classify_pipeline_error(e: anyhow::Error) -> ExtractError {
    // ffmpeg's `input()` surfaces HTTP status as strings like "Server returned
    // 404 Not Found". Match on the full error chain.
    let msg = format!("{e:#}");
    if msg.contains("404 Not Found") {
        ExtractError::UpstreamNotFound(msg)
    } else {
        ExtractError::Internal(msg)
    }
}

#[derive(Deserialize)]
pub struct ExtractRequest {
    pub video_path: String,
    #[serde(default)]
    pub known_plates: Vec<String>,
    pub plate_regex: Option<String>,
    pub interval_secs: Option<f64>,
    pub min_agreements: Option<usize>,
    pub ocr_min_conf: Option<f32>,
    pub known_distance: Option<usize>,
    pub min_vote_gap_secs: Option<f64>,
}

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Known,
    Unknown,
    Inconclusive,
    None,
}

#[derive(Serialize)]
pub struct ExtractResponse {
    pub status: Status,
    pub plate: Option<String>,
    pub agreements: usize,
    pub frames_scanned: usize,
    pub elapsed_ms: u64,
}

type Shared = Arc<Mutex<Pipeline>>;

pub async fn run(addr: &str, pipeline: Pipeline) -> Result<()> {
    let state: Shared = Arc::new(Mutex::new(pipeline));
    let app = Router::new()
        .route("/extract", post(handle_extract))
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_extract(
    State(pipeline): State<Shared>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, (StatusCode, String)> {
    let t0 = std::time::Instant::now();
    let min_agreements_cfg = req.min_agreements.unwrap_or(2);

    let result = tokio::task::spawn_blocking(move || -> Result<_, ExtractError> {
        let mut opts = PipelineOptions::default();
        opts.known_plates = req.known_plates;
        opts.min_agreements = min_agreements_cfg;
        if let Some(d) = req.known_distance {
            opts.known_distance = d;
        }
        if let Some(i) = req.interval_secs {
            opts.interval_secs = i;
        }
        if let Some(c) = req.ocr_min_conf {
            opts.ocr_min_conf = c;
        }
        if let Some(g) = req.min_vote_gap_secs {
            opts.min_vote_gap_secs = g;
        }
        if let Some(r) = req.plate_regex {
            opts.plate_regex = Some(
                regex::Regex::new(&r)
                    .map_err(|e| ExtractError::Client(format!("bad regex: {e}")))?,
            );
        }
        let mut pipe = pipeline
            .lock()
            .map_err(|e| ExtractError::Internal(format!("lock poisoned: {e}")))?;
        pipe.extract(Path::new(&req.video_path), &opts)
            .map_err(classify_pipeline_error)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")))?
    .map_err(ExtractError::into_response)?;

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    // Known plates win on any agreement count: nearest_known already enforces a
    // tight edit-distance bound, so a single-frame known match is trustworthy
    // even when the full min_agreements threshold isn't met.
    let (status, plate, agreements, frames_scanned) = match result {
        Some(d) if d.known => (
            Status::Known,
            Some(d.plate),
            d.agreements,
            d.frames_scanned,
        ),
        Some(d) if d.agreements >= min_agreements_cfg => (
            Status::Unknown,
            Some(d.plate),
            d.agreements,
            d.frames_scanned,
        ),
        Some(d) => (
            Status::Inconclusive,
            Some(d.plate),
            d.agreements,
            d.frames_scanned,
        ),
        None => (Status::None, None, 0, 0),
    };
    Ok(Json(ExtractResponse {
        status,
        plate,
        agreements,
        frames_scanned,
        elapsed_ms,
    }))
}
