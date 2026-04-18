use std::path::Path;

use anyhow::{anyhow, Result};
use ffmpeg_next as ffmpeg;
use ffmpeg::format::{context::Input, input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::util::rational::Rational;

const AV_TIME_BASE: f64 = 1_000_000.0;

pub fn init() -> Result<()> {
    ffmpeg::init().map_err(Into::into)
}

/// A reusable video decoder that can seek to arbitrary timestamps.
///
/// Seeking is not frame-accurate — ffmpeg jumps to the nearest keyframe at or
/// before the target, then we decode forward until we get a frame. For sparse
/// sampling (hundreds of ms apart) that's fine.
pub struct Decoder {
    ictx: Input,
    decoder: ffmpeg::decoder::Video,
    scaler: Scaler,
    stream_index: usize,
    stream_time_base: Rational,
    duration_secs: f64,
    /// PTS (in seconds) of the last frame we handed out. Used to skip the
    /// seek when the caller walks forward monotonically.
    last_frame_secs: Option<f64>,
}

impl Decoder {
    pub fn open(path: &Path) -> Result<Self> {
        let ictx = input(&path)?;
        let stream = ictx
            .streams()
            .best(Type::Video)
            .ok_or_else(|| anyhow!("no video stream"))?;
        let stream_index = stream.index();
        let stream_time_base = stream.time_base();

        let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        let decoder = decoder_ctx.decoder().video()?;

        let scaler = Scaler::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )?;

        let duration_secs = ictx.duration() as f64 / AV_TIME_BASE;

        Ok(Self {
            ictx,
            decoder,
            scaler,
            stream_index,
            stream_time_base,
            duration_secs,
            last_frame_secs: None,
        })
    }

    pub fn duration_secs(&self) -> f64 {
        self.duration_secs
    }

    pub fn width(&self) -> u32 {
        self.decoder.width()
    }

    pub fn height(&self) -> u32 {
        self.decoder.height()
    }

    /// Decode the first video frame at or after `timestamp_secs`.
    ///
    /// Seeks to the nearest keyframe ≤ target, then decodes forward, discarding
    /// frames whose PTS is before the target.
    pub fn frame_at(&mut self, timestamp_secs: f64) -> Result<image::RgbImage> {
        let tb_num = self.stream_time_base.numerator() as f64;
        let tb_den = self.stream_time_base.denominator() as f64;
        let pts_to_secs = |pts: i64| (pts as f64) * tb_num / tb_den;

        // Only seek when moving backwards or jumping far ahead of where the
        // decoder already is. Monotonic forward scans just decode forward.
        let need_seek = match self.last_frame_secs {
            None => true,
            Some(last) => timestamp_secs < last,
        };
        if need_seek {
            let seek_ts = (timestamp_secs * AV_TIME_BASE) as i64;
            self.ictx.seek(seek_ts, ..seek_ts)?;
            self.decoder.flush();
            self.last_frame_secs = None;
        }

        let stream_index = self.stream_index;
        let scaler = &mut self.scaler;
        let decoder = &mut self.decoder;
        let last_frame_secs = &mut self.last_frame_secs;
        let mut last_good: Option<image::RgbImage> = None;

        for (s, packet) in self.ictx.packets() {
            if s.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet)?;
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let Some(pts) = decoded.pts() else { continue };
                let frame_secs = pts_to_secs(pts);
                if frame_secs + 1e-3 >= timestamp_secs {
                    *last_frame_secs = Some(frame_secs);
                    return scale_to_image(scaler, &decoded);
                }
                *last_frame_secs = Some(frame_secs);
                last_good = Some(scale_to_image(scaler, &decoded)?);
            }
        }

        decoder.send_eof()?;
        let mut decoded = Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            if let Some(pts) = decoded.pts() {
                *last_frame_secs = Some(pts_to_secs(pts));
            }
            last_good = Some(scale_to_image(scaler, &decoded)?);
        }

        last_good.ok_or_else(|| anyhow!("no frame decoded at t={timestamp_secs}s"))
    }
}

fn scale_to_image(scaler: &mut Scaler, frame: &Video) -> Result<image::RgbImage> {
    let mut rgb = Video::empty();
    scaler.run(frame, &mut rgb)?;
    video_to_image(&rgb)
}

fn video_to_image(frame: &Video) -> Result<image::RgbImage> {
    let w = frame.width();
    let h = frame.height();
    let stride = frame.stride(0);
    let data = frame.data(0);

    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        let row_start = y * stride;
        let row_end = row_start + (w as usize) * 3;
        buf.extend_from_slice(&data[row_start..row_end]);
    }

    image::RgbImage::from_raw(w, h, buf).ok_or_else(|| anyhow!("image buffer size mismatch"))
}
