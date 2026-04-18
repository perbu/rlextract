use std::path::Path;

use anyhow::{anyhow, Result};
use image::RgbImage;
use ndarray::{Array4, ArrayView2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub const INPUT_SIZE: u32 = 640;

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub conf: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectorOptions {
    pub conf_threshold: f32,
    pub nms_iou: f32,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            conf_threshold: 0.25,
            nms_iou: 0.45,
        }
    }
}

pub struct Detector {
    session: Session,
    input_name: String,
    opts: DetectorOptions,
}

impl Detector {
    pub fn new(model_path: &Path, opts: DetectorOptions) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("ort: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("ort: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("ort: {e}"))?;
        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?
            .name()
            .to_string();
        Ok(Self {
            session,
            input_name,
            opts,
        })
    }

    /// Run a YOLOv8/v11 single-class plate detector on `img`.
    ///
    /// Expects an ONNX with input `[1,3,INPUT_SIZE,INPUT_SIZE]` f32 RGB in [0,1],
    /// and output `[1,5,N]` where rows are `[cx, cy, w, h, conf]` in input-space.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<BBox>> {
        let (tensor, scale, pad_x, pad_y) = letterbox(img, INPUT_SIZE);
        let input_value = Tensor::from_array(tensor)?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input_value])?;
        let (_, output) = outputs.iter().next().ok_or_else(|| anyhow!("no outputs"))?;
        let view = output.try_extract_array::<f32>()?;

        let shape = view.shape().to_vec();
        if shape.len() != 3 || shape[1] != 5 {
            return Err(anyhow!(
                "unexpected output shape {:?}, expected [1,5,N]",
                shape
            ));
        }
        let n = shape[2];
        let view = view.to_shape((5, n))?;
        let rows: ArrayView2<f32> = view.view();

        let mut boxes = Vec::new();
        for i in 0..n {
            let conf = rows[[4, i]];
            if conf < self.opts.conf_threshold {
                continue;
            }
            let cx = rows[[0, i]];
            let cy = rows[[1, i]];
            let w = rows[[2, i]];
            let h = rows[[3, i]];
            // undo letterbox: input-space → original-image space
            let x = (cx - w / 2.0 - pad_x) / scale;
            let y = (cy - h / 2.0 - pad_y) / scale;
            let w = w / scale;
            let h = h / scale;
            boxes.push(BBox { x, y, w, h, conf });
        }

        Ok(nms(boxes, self.opts.nms_iou))
    }
}

fn letterbox(img: &RgbImage, size: u32) -> (Array4<f32>, f32, f32, f32) {
    let (w, h) = img.dimensions();
    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);

    let pad_x = (size - new_w) as f32 / 2.0;
    let pad_y = (size - new_h) as f32 / 2.0;
    let pad_x_i = pad_x as usize;
    let pad_y_i = pad_y as usize;

    // Gray padding (0.5) matches Ultralytics default.
    let mut arr = Array4::<f32>::from_elem((1, 3, size as usize, size as usize), 0.5);
    let raw = resized.as_raw();
    let row_len = new_w as usize * 3;
    for y in 0..new_h as usize {
        let row = &raw[y * row_len..(y + 1) * row_len];
        let oy = y + pad_y_i;
        for (x, px) in row.chunks_exact(3).enumerate() {
            let ox = x + pad_x_i;
            arr[[0, 0, oy, ox]] = px[0] as f32 / 255.0;
            arr[[0, 1, oy, ox]] = px[1] as f32 / 255.0;
            arr[[0, 2, oy, ox]] = px[2] as f32 / 255.0;
        }
    }
    (arr, scale, pad_x, pad_y)
}

fn iou(a: &BBox, b: &BBox) -> f32 {
    let (ax2, ay2) = (a.x + a.w, a.y + a.h);
    let (bx2, by2) = (b.x + b.w, b.y + b.h);
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn nms(mut boxes: Vec<BBox>, iou_thresh: f32) -> Vec<BBox> {
    boxes.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    while !boxes.is_empty() {
        let best = boxes.remove(0);
        boxes.retain(|b| iou(&best, b) < iou_thresh);
        keep.push(best);
    }
    keep
}
