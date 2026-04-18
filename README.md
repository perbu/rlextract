# rlextract

Offline license-plate extraction from mp4 clips. Given a short video (e.g. a
Frigate event clip), it samples frames, runs a YOLO-family plate detector,
OCRs each detection, and returns a decision once N frames agree on the same
plate string. Intended to be called from Home Assistant to gate automations on
known vehicles.

The crate is camera- and country-agnostic. Any site-specific logic (known
plates, plate-format regex, confidence thresholds) is supplied per request.

## Requirements

- Rust stable
- FFmpeg development headers (`libavformat`, `libavcodec`, `libswscale`, `libavutil`)
- A YOLOv8/v11-format ONNX plate detector (`[1,5,N]` output)
- A fast-plate-ocr ONNX recogniser (`[1, slots, classes]` output)

Models are not bundled. Download them yourself; see below.

## Build

### macOS

```sh
brew install ffmpeg
cargo build --release
```

### Raspberry Pi 5 (build on-device)

```sh
sudo apt install build-essential pkg-config \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libclang-dev
cargo build --release
```

### Cross-compile from macOS

```sh
cargo install cargo-zigbuild
brew install zig
cargo zigbuild --release --target aarch64-unknown-linux-gnu
```

If ort complains about a missing `libonnxruntime.so` on the Pi, set
`ORT_DYLIB_PATH` to a manually downloaded `aarch64` runtime.

## Models

```sh
# Plate detector (~10 MB)
curl -LO https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.onnx

# OCR (~5 MB)
curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx
```

## CLI

```sh
# Single frame → annotated PNG (sanity check)
rlextract detect clip.mp4 12.0 \
    -m license-plate-finetune-v1n.onnx -o detect.png

# End-to-end sweep with voting
rlextract run clip.mp4 \
    --detector license-plate-finetune-v1n.onnx \
    --ocr cct_s_v2_global.onnx \
    --plate-regex '^[A-Z]{2}\d{5}$' \
    --known EL67751
```

## HTTP service

```sh
rlextract serve --addr 127.0.0.1:8787 \
    --detector license-plate-finetune-v1n.onnx \
    --ocr cct_s_v2_global.onnx
```

`video_path` accepts either a local filesystem path or an `http(s)://` URL
(handled by ffmpeg's native HTTP protocol — one streaming GET per clip, no
temporary files).

```
POST /extract
{
  "video_path": "http://frigate:5000/api/events/<id>/clip.mp4",
  "known_plates": ["EL67751"],
  "plate_regex": "^[A-Z]{2}\\d{5}$"
}

→ { "status": "known"|"unknown"|"inconclusive"|"none",
    "plate": "EL67751",
    "agreements": 2,
    "frames_scanned": 7,
    "elapsed_ms": 498 }
```

`GET /healthz` returns `ok`.

## systemd

```ini
[Unit]
Description=rlextract
After=network.target

[Service]
ExecStart=/usr/local/bin/rlextract serve \
    --addr 127.0.0.1:8787 \
    --detector /etc/rlextract/license-plate-finetune-v1n.onnx \
    --ocr /etc/rlextract/cct_s_v2_global.onnx
Restart=always
User=rlextract

[Install]
WantedBy=multi-user.target
```

## Home Assistant

```yaml
rest_command:
  rlextract:
    url: "http://127.0.0.1:8787/extract"
    method: POST
    content_type: "application/json"
    payload: >-
      {"video_path": "{{ clip_url }}",
       "known_plates": {{ known | tojson }},
       "plate_regex": "^[A-Z]{2}\\d{5}$"}
    timeout: 15
```

Branch on `r.content.status` in your automation.

## License

Apache-2.0. See `LICENSE.md`.
