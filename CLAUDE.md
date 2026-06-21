# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AiSwap is a repository containing the Deep-Live-Cam project, an AI-powered real-time face-swapping application. The project supports multiple modes:
- Image-to-image face swapping
- Image-to-video face swapping
- Real-time webcam face swapping with virtual camera output
- Remote GPU processing via Google Colab with ZMQ over Tailscale

## Repository Structure

```
projects/Deep-Live-Cam/
  ├── run.py                    # Entry point → calls modules.core.run()
  ├── modules/
  │   ├── core.py              # Main orchestration: CLI parsing, workflow control
  │   ├── globals.py           # Global state/config (paths, execution providers, remote settings)
  │   ├── ui.py                # CustomTkinter GUI
  │   ├── processors/frame/    # Frame processing pipeline
  │   │   ├── face_swapper.py       # InsightFace-based local swap
  │   │   ├── face_enhancer.py      # GFPGAN enhancement
  │   │   └── remote_processor.py   # ZMQ client for Colab GPU
  │   ├── gpu_processing.py    # GPU detection and optimization
  │   ├── onnx_optimize.py     # ONNX model optimization utilities
  │   └── utilities.py         # Video/image helpers (ffmpeg wrappers)
  ├── models/                  # Pre-trained models (downloaded separately)
  ├── google-colab/            # Colab notebooks for remote processing
  └── benchmark_pipeline.py    # Performance benchmarking tool
```

## Development Commands

### Environment Setup
```bash
# Navigate to Deep-Live-Cam directory
cd projects/Deep-Live-Cam

# Create/activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**GUI Mode (default):**
```bash
python run.py
```

**CLI Mode (headless):**
```bash
# Image swap
python run.py -s source.jpg -t target.jpg -o output.jpg

# Video swap
python run.py -s source.jpg -t video.mp4 -o output.mp4 --execution-provider cuda

# Remote processing (requires Colab instance)
python run.py --frame-processor remote_processor --remote-host <tailscale-ip>

# Live webcam mode with remote processing
python run.py --live --frame-processor remote_processor --remote-host <tailscale-ip> \
  --source source.jpg --camera-index 0 --virtual-camera "OBS Virtual Camera"
```

**Execution Providers:**
- `cpu` - CPU-only (slow but universal)
- `cuda` - NVIDIA GPU (requires CUDA Toolkit 11.8 + onnxruntime-gpu)
- `directml` - Windows GPU (AMD/Intel/NVIDIA via DirectML)
- `coreml` - Apple Silicon

### Benchmarking
```bash
python benchmark_pipeline.py
```
Captures 200 frames from webcam and measures per-stage timing (detection, ONNX swap, paste-back).

## Architecture Notes

### Core Pipeline Flow
1. **Argument Parsing** (`core.py:parse_args`): Decodes CLI args → `modules.globals`
2. **Pre-checks** (`core.py:pre_check`): Validates Python 3.9+, ffmpeg presence
3. **Resource Limiting** (`core.py:limit_resources`): Sets memory caps via OS APIs
4. **Processor Selection** (`processors/frame/core.py`): Dynamically imports frame processors based on `--frame-processor` flag
5. **Execution**:
   - **Image mode**: Direct frame processing + save
   - **Video mode**: Extract frames → process batch → ffmpeg encode → restore audio
   - **Live mode**: OpenCV capture → remote ZMQ processing → pyvirtualcam output

### Remote Processing (ZMQ Architecture)
- **Client** (`remote_processor.py`): Windows machine with physical webcam
  - Sends source face image once at startup (via `push_addr` on port 5555)
  - Sends target image for processing (via `push_addr_two` on port 5556)
  - Receives processed result (via `pull_addr` on port 5557)
- **Server** (Colab notebook): Runs ZMQ REP sockets to receive frames, perform GPU swap, return results
- **Transport**: Tailscale VPN for secure tunneling
- **Protocol**: Single multipart ZMQ messages (metadata JSON + raw image bytes)
  - **Old protocol** (chunked): 3 round trips per image (metadata→ACK, chunks→ACK each, END→ACK) × 3 images = **9 total round trips**
  - **New protocol** (multipart): 1 round trip per image (multipart→ACK) × 3 images = **3 total round trips**
  - **Latency reduction**: ~67% fewer network round trips
  - **Implementation**: `send_multipart([metadata_json, raw_bytes])` + single ACK
- **Key Optimizations**:
  - Source face caching: Colab analyzes source face once, caches the detected face object for subsequent swaps
  - Raw uint8 images: No compression overhead, ~660 KB for 624×352 result
  - InsightFace warm-up: Models initialized once during Colab startup
- **Key Config** (`globals.py`):
  - `push_addr`, `push_addr_two`, `pull_addr` - ZMQ endpoints (ports 5555, 5556, 5557)
  - `live_width`, `live_height`, `live_fps` - Stream resolution/framerate
  - `gof`, `bitrate`, `maxrate`, `bufsize` - ffmpeg encoding params (live mode only)

### Frame Processors
Frame processors implement this interface (see `processors/frame/core.py`):
- `pre_check() -> bool` - Validate dependencies/models
- `pre_start() -> bool` - Validate runtime conditions
- `process_image(source, target, output)` - Single image
- `process_video(source, frame_paths)` - Batch frames
- `process_frames(source, frames, progress)` - Called internally by process_video

**Available processors:**
- `face_swapper` - Local InsightFace swap (inswapper_128_fp16.onnx)
- `face_enhancer` - GFPGAN upscaling (GFPGANv1.4.pth)
- `remote_processor` - ZMQ-based Colab offload

### Model Files
Required in `models/` directory:
- `inswapper_128_fp16.onnx` - Face swapping model (277 MB)
- `GFPGANv1.4.pth` - Face enhancement model (348 MB)

Downloaded from: https://huggingface.co/hacksider/deep-live-cam

### GPU Optimizations
- `modules/gpu_processing.py`: Auto-detects CUDA/ROCm/DirectML
- `modules/onnx_optimize.py`: Applies graph optimizations to ONNX models
- `OMP_NUM_THREADS=1` set before torch import (doubles CUDA perf on single-threaded workloads)

### Global State Pattern
All runtime config lives in `modules/globals.py` (not best practice but matches upstream codebase):
- Paths: `source_path`, `target_path`, `output_path`
- Execution: `execution_providers`, `execution_threads`, `max_memory`
- Processing: `frame_processors`, `many_faces`, `keep_fps`, `keep_audio`
- Remote: `push_addr`, `pull_addr`, `remote_face_enhancer`
- Live: `camera_index`, `virtual_camera`, `live_width/height/fps`

## Important Context

### Upstream Relationship
- **Base project**: [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) (originally roop-cam, derived from s0md3v/roop)
- **Current status**: Working fork with custom optimizations and remote processing features
- Recent commits show major refactoring, ZMQ architecture fixes, and Tailscale integration

### Key Dependencies
- **InsightFace**: Face detection/recognition (buffalo_l model)
- **ONNX Runtime**: Model inference with GPU acceleration
- **OpenCV**: Video/image I/O, frame manipulation
- **ffmpeg**: Video encoding/decoding, audio handling
- **ZMQ (PyZMQ)**: Remote processing transport
- **CustomTkinter**: Modern GUI framework
- **pyvirtualcam**: Virtual camera output (Windows only, for live mode)

### Performance Considerations
- Detection runs every 3rd frame by default (cached face tracking reduces overhead)
- `_fast_paste_back` in `face_swapper.py` performs in-place writes (no frame.copy())
- Benchmark targets ~60 FPS on 1080p with CUDA providers
- **Remote processing latency** (image mode):
  - Network round trips dominate latency (not payload size)
  - Optimized protocol: 3 round trips per swap vs. 9 in old chunked protocol
  - Typical Tailscale RTT: 20-50ms → total transport ~60-150ms (3 × RTT)
  - GPU inference: ~50-100ms (T4 Colab GPU)
  - **Total latency**: ~110-250ms per image (down from ~230-550ms with chunked protocol)

### Code Patterns
- **No type hints**: Legacy codebase from roop, minimal typing
- **Global state**: `modules.globals` is imported everywhere
- **UI/Core separation**: `core.py` can run headless or with `ui.py` tkinter window
- **Deprecation warnings**: CLI args handle legacy flags from roop/roop-cam

## Development Workflow

1. **Making changes to core logic**: Edit `modules/core.py` or processor files
2. **Adding frame processors**: Create new module in `processors/frame/`, implement interface
3. **Modifying remote protocol**: Update both `remote_processor.py` (client) and Colab notebook (server)
4. **Testing local changes**: Use `benchmark_pipeline.py` for performance validation
5. **Testing remote mode**: Requires running Colab notebook + Tailscale setup

## Common Gotchas

- **NSFW check disabled**: `modules.globals.nsfw = True` hardcoded (line 144 in core.py) to allow CPU-only remote clients
- **Windows virtual camera**: Requires separate driver installation (e.g., OBS Virtual Camera)
- **CUDA memory**: Default `--max-memory 16` may need tuning for low-VRAM GPUs
- **ffmpeg required**: Pre-check will fail without ffmpeg in PATH
- **Model auto-download**: First run downloads ~300MB of InsightFace models to `.insightface/`
