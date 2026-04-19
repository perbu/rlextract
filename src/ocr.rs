use std::path::Path;

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, RgbImage};
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, TensorElementType};

/// Default alphabet for fast-plate-ocr global models.
/// Last entry is the blank/padding token.
pub const DEFAULT_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

#[derive(Debug, Clone, Copy)]
enum Layout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum InputDtype {
    U8,
    F32,
}

pub struct Recognizer {
    session: Session,
    input_name: String,
    layout: Layout,
    dtype: InputDtype,
    h: u32,
    w: u32,
    channels: u32,
    alphabet: Vec<char>,
}

impl Recognizer {
    pub fn new(model_path: &Path, alphabet: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("ort: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("ort: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("ort: {e}"))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?;
        let input_name = input.name().to_string();
        let shape = input
            .dtype()
            .tensor_shape()
            .ok_or_else(|| anyhow!("input is not a tensor"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 4 {
            return Err(anyhow!("expected 4D input, got {:?}", dims));
        }
        let elem_ty = input
            .dtype()
            .tensor_type()
            .ok_or_else(|| anyhow!("cannot read input element type"))?;
        let dtype = match elem_ty {
            TensorElementType::Uint8 => InputDtype::U8,
            TensorElementType::Float32 => InputDtype::F32,
            other => {
                return Err(anyhow!(
                    "unsupported input element type {:?}, want Uint8 or Float32",
                    other
                ))
            }
        };

        // Heuristic: if dim[1] is 1 or 3, assume NCHW; if dim[3] is 1 or 3, NHWC.
        let (layout, h, w, c) = match (dims[1], dims[3]) {
            (1 | 3, _) => (Layout::Nchw, dims[2] as u32, dims[3] as u32, dims[1] as u32),
            (_, 1 | 3) => (Layout::Nhwc, dims[1] as u32, dims[2] as u32, dims[3] as u32),
            _ => {
                return Err(anyhow!(
                    "can't infer layout from input dims {:?}",
                    dims
                ))
            }
        };

        eprintln!(
            "ocr model: {:?} {:?} {}x{}x{}ch, input='{}'",
            layout, dtype, h, w, c, input_name
        );

        Ok(Self {
            session,
            input_name,
            layout,
            dtype,
            h,
            w,
            channels: c,
            alphabet: alphabet.chars().collect(),
        })
    }

    /// Recognize text in `crop` (already cropped to the plate region).
    pub fn recognize(&mut self, crop: &RgbImage) -> Result<(String, f32)> {
        let outputs = match self.dtype {
            InputDtype::F32 => {
                let input_value = Tensor::from_array(self.preprocess_f32(crop))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => input_value])?
            }
            InputDtype::U8 => {
                let input_value = Tensor::from_array(self.preprocess_u8(crop))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => input_value])?
            }
        };
        let (_, out) = outputs.iter().next().ok_or_else(|| anyhow!("no outputs"))?;
        let view = out.try_extract_array::<f32>()?;
        let shape = view.shape().to_vec();
        if shape.len() != 3 {
            return Err(anyhow!("unexpected OCR output shape {:?}", shape));
        }
        let slots = shape[1];
        let classes = shape[2];
        let flat = view.to_shape((slots, classes))?;

        // Detect whether the graph already includes a softmax. fast-plate-ocr
        // exports do; a plain classification head would emit raw logits.
        let already_softmax = {
            let row0 = flat.row(0);
            let mn = row0.iter().cloned().fold(f32::INFINITY, f32::min);
            let sum: f32 = row0.iter().sum();
            mn >= 0.0 && (sum - 1.0).abs() < 1e-2
        };

        let mut text = String::new();
        let mut conf_sum = 0.0f32;
        let mut conf_count = 0usize;
        for s in 0..slots {
            let row = flat.row(s);
            let (best_idx, best_raw) = row
                .iter()
                .cloned()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |acc, x| if x.1 > acc.1 { x } else { acc });
            let best_p = if already_softmax {
                best_raw
            } else {
                let denom: f32 = row.iter().map(|v| (v - best_raw).exp()).sum();
                1.0 / denom
            };
            if best_idx < self.alphabet.len() {
                let ch = self.alphabet[best_idx];
                if ch != '_' {
                    text.push(ch);
                    conf_sum += best_p;
                    conf_count += 1;
                }
            }
        }
        let mean_conf = if conf_count > 0 {
            conf_sum / conf_count as f32
        } else {
            0.0
        };
        Ok((text, mean_conf))
    }

    fn preprocess_u8(&self, img: &RgbImage) -> Array4<u8> {
        let resized = image::imageops::resize(img, self.w, self.h, FilterType::Triangle);
        let (h, w, c) = (self.h as usize, self.w as usize, self.channels as usize);
        let shape = match self.layout {
            Layout::Nchw => (1, c, h, w),
            Layout::Nhwc => (1, h, w, c),
        };
        let mut arr = Array4::<u8>::zeros(shape);
        let raw = resized.as_raw();

        // Fast path: the most common production case is NHWC RGB, which matches
        // the image crate's native HWC packing exactly — single memcpy.
        if self.channels == 3 && matches!(self.layout, Layout::Nhwc) {
            arr.as_slice_mut().expect("contiguous").copy_from_slice(raw);
            return arr;
        }

        for y in 0..h {
            for x in 0..w {
                let off = (y * w + x) * 3;
                let (r, g, b) = (raw[off], raw[off + 1], raw[off + 2]);
                match (self.channels, self.layout) {
                    (1, Layout::Nchw) => arr[[0, 0, y, x]] = rgb_to_gray_u8(r, g, b),
                    (1, Layout::Nhwc) => arr[[0, y, x, 0]] = rgb_to_gray_u8(r, g, b),
                    (3, Layout::Nchw) => {
                        arr[[0, 0, y, x]] = r;
                        arr[[0, 1, y, x]] = g;
                        arr[[0, 2, y, x]] = b;
                    }
                    _ => {}
                }
            }
        }
        arr
    }

    fn preprocess_f32(&self, img: &RgbImage) -> Array4<f32> {
        self.preprocess_u8(img).mapv(|v| v as f32 / 255.0)
    }
}

#[inline]
fn rgb_to_gray_u8(r: u8, g: u8, b: u8) -> u8 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8
}

pub fn crop_bbox(img: &RgbImage, x: f32, y: f32, w: f32, h: f32, pad: f32) -> RgbImage {
    let (iw, ih) = img.dimensions();
    let pad_x = w * pad;
    let pad_y = h * pad;
    let x0 = ((x - pad_x).max(0.0)) as u32;
    let y0 = ((y - pad_y).max(0.0)) as u32;
    let x1 = ((x + w + pad_x).min(iw as f32)) as u32;
    let y1 = ((y + h + pad_y).min(ih as f32)) as u32;
    image::imageops::crop_imm(img, x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
        .to_image()
}
