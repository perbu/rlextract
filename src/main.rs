use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use rlextract::{decode, detect, ocr, pipeline, serve};

#[derive(Parser)]
#[command(about = "License-plate extraction from mp4 files")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Decode a single frame at a given timestamp.
    Grab {
        input: PathBuf,
        #[arg(default_value_t = 0.0)]
        t: f64,
        #[arg(short, long, default_value = "frame.png")]
        out: PathBuf,
    },
    /// Run the plate detector on a single frame and save an annotated PNG.
    Detect {
        input: PathBuf,
        /// ONNX model (YOLOv8/v11, single-class plate)
        #[arg(short, long)]
        model: PathBuf,
        #[arg(default_value_t = 0.0)]
        t: f64,
        #[arg(short, long, default_value = "detect.png")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.25)]
        conf: f32,
    },
    /// HTTP service exposing POST /extract for Home Assistant.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        #[arg(long)]
        detector: PathBuf,
        #[arg(long)]
        ocr: PathBuf,
        #[arg(long, default_value = ocr::DEFAULT_ALPHABET)]
        alphabet: String,
        #[arg(long, default_value_t = 0.25)]
        det_conf: f32,
    },
    /// Sweep the full mp4 and return the first plate that N frames agree on.
    Run {
        input: PathBuf,
        #[arg(long)]
        detector: PathBuf,
        #[arg(long)]
        ocr: PathBuf,
        #[arg(long, default_value = ocr::DEFAULT_ALPHABET)]
        alphabet: String,
        #[arg(long, default_value_t = 0.5)]
        interval: f64,
        #[arg(long, default_value_t = 2)]
        min_agreements: usize,
        #[arg(long, default_value_t = 0.25)]
        det_conf: f32,
        #[arg(long, default_value_t = 0.3)]
        ocr_min_conf: f32,
        /// Minimum seconds between two frames agreeing on the same plate.
        #[arg(long, default_value_t = 2.0)]
        min_vote_gap: f64,
        /// Optional regex the plate must match (e.g. '^[A-Z]{2}\d{5}$' for NO).
        #[arg(long)]
        plate_regex: Option<String>,
        /// Known plate (repeatable). OCR reads within `--known-distance` edits
        /// of any of these early-exit as the canonical known plate.
        #[arg(long = "known", value_name = "PLATE")]
        known: Vec<String>,
        /// File with one known plate per line (comments with '#' and blank
        /// lines ignored). Merged with any `--known` flags.
        #[arg(long)]
        known_file: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        known_distance: usize,
    },
    /// Full pipeline: decode frame, detect plate(s), OCR each, print strings.
    Recognize {
        input: PathBuf,
        #[arg(long)]
        detector: PathBuf,
        #[arg(long)]
        ocr: PathBuf,
        #[arg(default_value_t = 0.0)]
        t: f64,
        #[arg(long, default_value = ocr::DEFAULT_ALPHABET)]
        alphabet: String,
        #[arg(long, default_value_t = 0.25)]
        det_conf: f32,
        /// Fraction of bbox size to pad around the crop before OCR.
        #[arg(long, default_value_t = 0.1)]
        pad: f32,
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Dev aid: sample frames at fixed intervals, write each to a directory.
    Dump {
        input: PathBuf,
        #[arg(short, long, default_value_t = 0.5)]
        interval: f64,
        #[arg(short, long, default_value = "frames")]
        out_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    decode::init()?;

    match cli.cmd {
        Cmd::Serve {
            addr,
            detector,
            ocr: ocr_model,
            alphabet,
            det_conf,
        } => {
            let det = detect::Detector::new(
                &detector,
                detect::DetectorOptions {
                    conf_threshold: det_conf,
                    ..Default::default()
                },
            )?;
            let rec = ocr::Recognizer::new(&ocr_model, &alphabet)?;
            let pipe = pipeline::Pipeline::new(det, rec);
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(serve::run(&addr, pipe))?;
        }
        Cmd::Grab { input, t, out } => {
            let mut dec = decode::Decoder::open(&input)?;
            let img = dec.frame_at(t)?;
            println!("decoded {}x{} at t={}s", img.width(), img.height(), t);
            img.save(&out)?;
            println!("wrote {}", out.display());
        }
        Cmd::Detect {
            input,
            model,
            t,
            out,
            conf,
        } => {
            let mut dec = decode::Decoder::open(&input)?;
            let img = dec.frame_at(t)?;
            let mut det = detect::Detector::new(
                &model,
                detect::DetectorOptions {
                    conf_threshold: conf,
                    ..Default::default()
                },
            )?;
            let boxes = det.detect(&img)?;
            println!("{} detections at t={}s", boxes.len(), t);
            for b in &boxes {
                println!(
                    "  conf={:.3} xywh=({:.0},{:.0},{:.0},{:.0})",
                    b.conf, b.x, b.y, b.w, b.h
                );
            }
            let annotated = annotate(&img, &boxes);
            annotated.save(&out)?;
            println!("wrote {}", out.display());
        }
        Cmd::Run {
            input,
            detector,
            ocr: ocr_model,
            alphabet,
            interval,
            min_agreements,
            det_conf,
            ocr_min_conf,
            min_vote_gap,
            plate_regex,
            known,
            known_file,
            known_distance,
        } => {
            let det = detect::Detector::new(
                &detector,
                detect::DetectorOptions {
                    conf_threshold: det_conf,
                    ..Default::default()
                },
            )?;
            let rec = ocr::Recognizer::new(&ocr_model, &alphabet)?;
            let mut pipe = pipeline::Pipeline::new(det, rec);
            let plate_regex = plate_regex
                .as_deref()
                .map(regex::Regex::new)
                .transpose()?;
            let mut known_plates = known;
            if let Some(path) = &known_file {
                let body = std::fs::read_to_string(path)?;
                for line in body.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    known_plates.push(line.to_string());
                }
            }
            let opts = pipeline::PipelineOptions {
                interval_secs: interval,
                min_agreements,
                ocr_min_conf,
                min_vote_gap_secs: min_vote_gap,
                plate_regex,
                known_plates,
                known_distance,
                ..Default::default()
            };
            let t0 = std::time::Instant::now();
            match pipe.extract(&input, &opts)? {
                Some(d) if d.agreements >= min_agreements => {
                    println!(
                        "{} {} (agreements={}, frames={}, t={:?}, elapsed={:.2}s)",
                        if d.known { "KNOWN" } else { "UNKNOWN" },
                        d.plate,
                        d.agreements,
                        d.frames_scanned,
                        d.timestamps,
                        t0.elapsed().as_secs_f32()
                    );
                }
                Some(d) => {
                    println!(
                        "INCONCLUSIVE best='{}' known={} agreements={} frames={} (needed {})",
                        d.plate, d.known, d.agreements, d.frames_scanned, min_agreements
                    );
                }
                None => {
                    println!("NO PLATE DETECTED");
                }
            }
        }
        Cmd::Recognize {
            input,
            detector,
            ocr: ocr_model,
            t,
            alphabet,
            det_conf,
            pad,
            out,
        } => {
            let mut dec = decode::Decoder::open(&input)?;
            let img = dec.frame_at(t)?;
            let mut det = detect::Detector::new(
                &detector,
                detect::DetectorOptions {
                    conf_threshold: det_conf,
                    ..Default::default()
                },
            )?;
            let boxes = det.detect(&img)?;
            println!("{} detections at t={}s", boxes.len(), t);
            let mut rec = ocr::Recognizer::new(&ocr_model, &alphabet)?;
            for (i, b) in boxes.iter().enumerate() {
                let crop = ocr::crop_bbox(&img, b.x, b.y, b.w, b.h, pad);
                if let Some(p) = &out {
                    let stem = p.file_stem().unwrap_or_default().to_string_lossy();
                    let ext = p.extension().unwrap_or_default().to_string_lossy();
                    let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
                    let cp = parent.join(format!("{stem}_crop{i}.{ext}"));
                    crop.save(&cp)?;
                }
                let (text, conf) = rec.recognize(&crop)?;
                println!(
                    "  det_conf={:.3} plate='{}' ocr_conf={:.3}",
                    b.conf, text, conf
                );
            }
            if let Some(p) = &out {
                annotate(&img, &boxes).save(p)?;
            }
        }
        Cmd::Dump {
            input,
            interval,
            out_dir,
        } => {
            std::fs::create_dir_all(&out_dir)?;
            let mut dec = decode::Decoder::open(&input)?;
            let duration = dec.duration_secs();
            println!(
                "{}x{}, {:.2}s, sampling every {}s",
                dec.width(),
                dec.height(),
                duration,
                interval
            );

            let mut t = 0.0;
            let mut n = 0;
            while t < duration {
                match dec.frame_at(t) {
                    Ok(img) => {
                        let path = out_dir.join(format!("frame_{:06.2}s.png", t));
                        img.save(&path)?;
                        n += 1;
                    }
                    Err(e) => eprintln!("t={:.2}: {}", t, e),
                }
                t += interval;
            }
            println!("wrote {n} frames to {}", out_dir.display());
        }
    }
    Ok(())
}

fn annotate(img: &image::RgbImage, boxes: &[detect::BBox]) -> image::RgbImage {
    let mut out = img.clone();
    let color = image::Rgb([0u8, 255, 0]);
    for b in boxes {
        let rect = imageproc::rect::Rect::at(b.x as i32, b.y as i32)
            .of_size(b.w.max(1.0) as u32, b.h.max(1.0) as u32);
        imageproc::drawing::draw_hollow_rect_mut(&mut out, rect, color);
    }
    out
}
