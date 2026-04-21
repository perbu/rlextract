use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use regex::Regex;

use crate::decode::{Decoder, SweepAction};
use crate::detect::Detector;
use crate::ocr::{crop_bbox, Recognizer};

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    /// Sampling interval in seconds.
    pub interval_secs: f64,
    /// Number of frames that must produce the same plate string before we
    /// return it. 2 is usually enough; raise for safety-critical decisions.
    pub min_agreements: usize,
    /// OCR mean-confidence floor; results below this don't count toward a vote.
    pub ocr_min_conf: f32,
    /// Minimum plate string length that can enter a vote.
    pub min_plate_len: usize,
    /// Padding around the detected bbox before cropping for OCR.
    pub crop_pad: f32,
    /// Minimum time gap (seconds) between two frames agreeing on the same
    /// plate. Guards against correlated misreads from near-duplicate frames.
    pub min_vote_gap_secs: f64,
    /// Optional regex the plate string must match to count as a vote.
    /// Example for Norway: `^[A-Z]{2}\d{5}$`.
    pub plate_regex: Option<Regex>,
    /// Canonical plates that are "ours". OCR reads within `known_distance`
    /// edits of any of these get voted in as the canonical string — lets the
    /// pipeline early-exit on a known car even through minor misreads.
    pub known_plates: Vec<String>,
    /// Max Levenshtein distance for a known-plate match. 1 is usually right
    /// (tolerates a single ambiguous character like E vs F, 8 vs B).
    pub known_distance: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            interval_secs: 0.5,
            min_agreements: 2,
            ocr_min_conf: 0.3,
            min_plate_len: 4,
            crop_pad: 0.1,
            min_vote_gap_secs: 2.0,
            plate_regex: None,
            known_plates: Vec::new(),
            known_distance: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlateDecision {
    pub plate: String,
    pub agreements: usize,
    pub timestamps: Vec<f64>,
    pub frames_scanned: usize,
    /// True if `plate` matched one of the configured known plates.
    pub known: bool,
}

pub struct Pipeline {
    detector: Detector,
    recognizer: Recognizer,
}

impl Pipeline {
    pub fn new(detector: Detector, recognizer: Recognizer) -> Self {
        Self {
            detector,
            recognizer,
        }
    }

    /// Sweep the mp4, early-exit when `min_agreements` frames agree on a plate.
    /// Returns `None` if no plate met the bar before the video ended.
    pub fn extract(&mut self, mp4: &Path, opts: &PipelineOptions) -> Result<Option<PlateDecision>> {
        let mut dec = Decoder::open(mp4)?;

        let mut votes: HashMap<String, Vec<f64>> = HashMap::new();
        let mut known_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut frames_scanned = 0usize;
        let mut decision: Option<PlateDecision> = None;

        let detector = &mut self.detector;
        let recognizer = &mut self.recognizer;

        dec.sweep(opts.interval_secs, |t, frame| {
            frames_scanned += 1;
            let boxes = detector.detect(&frame)?;
            for b in &boxes {
                let crop = crop_bbox(&frame, b.x, b.y, b.w, b.h, opts.crop_pad);
                let (text, conf) = recognizer.recognize(&crop)?;
                if conf < opts.ocr_min_conf || text.len() < opts.min_plate_len {
                    continue;
                }
                // Snap to a known plate if within tolerance. Format regex is
                // applied to the raw OCR so we don't reject a known-plate
                // match on a one-char misread that happens to fail the regex.
                let (canonical, is_known) =
                    match nearest_known(&text, &opts.known_plates, opts.known_distance) {
                        Some(k) => (k, true),
                        None => {
                            if let Some(re) = &opts.plate_regex {
                                if !re.is_match(&text) {
                                    continue;
                                }
                            }
                            (text.clone(), false)
                        }
                    };
                if is_known {
                    known_set.insert(canonical.clone());
                }
                let stamps = votes.entry(canonical.clone()).or_default();
                // Skip votes too close in time to the last one for the same
                // plate — prevents correlated frames from rubber-stamping a
                // systematic misread.
                if let Some(&last_t) = stamps.last() {
                    if t - last_t < opts.min_vote_gap_secs {
                        continue;
                    }
                }
                stamps.push(t);
                if stamps.len() >= opts.min_agreements {
                    decision = Some(PlateDecision {
                        plate: canonical,
                        agreements: stamps.len(),
                        timestamps: stamps.clone(),
                        frames_scanned,
                        known: is_known,
                    });
                    return Ok(SweepAction::Stop);
                }
            }
            Ok(SweepAction::Continue)
        })?;

        if let Some(d) = decision {
            return Ok(Some(d));
        }

        let best = votes.into_iter().max_by_key(|(_, v)| v.len());
        Ok(best.map(|(plate, stamps)| {
            let known = known_set.contains(&plate);
            PlateDecision {
                plate,
                agreements: stamps.len(),
                timestamps: stamps,
                frames_scanned,
                known,
            }
        }))
    }
}

fn nearest_known(candidate: &str, known: &[String], max_dist: usize) -> Option<String> {
    if known.is_empty() {
        return None;
    }
    let mut best: Option<(usize, &String)> = None;
    for k in known {
        let d = strsim::levenshtein(candidate, k);
        if d > max_dist {
            continue;
        }
        if best.map_or(true, |(bd, _)| d < bd) {
            best = Some((d, k));
        }
    }
    best.map(|(_, k)| k.clone())
}
