# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJ0b2NfdmlzaWJsZSI6dHJ1ZSwiY29sbGFwc2VkX3NlY3Rpb25zIjpbInJ1bnRpbWUtYnVuZGxlIl0sIm5hbWUiOiJEZWVwX0xpdmVfQ2FtX1JlbW90ZV9CYXRjaC5pcHluYiJ9LCJrZXJuZWxzcGVjIjp7ImRpc3BsYXlfbmFtZSI6IlB5dGhvbiAzIiwibGFuZ3VhZ2UiOiJweXRob24iLCJuYW1lIjoicHl0aG9uMyJ9LCJsYW5ndWFnZV9pbmZvIjp7Im5hbWUiOiJweXRob24ifSwibmJmb3JtYXQiOjQsIm5iZm9ybWF0X21pbm9yIjo1fQ==

# %% [markdown] cell=0 id=title
"""MARKDOWN
# Deep-Live-Cam Remote — Colab batch processor

Self-contained, path-based photo/video batch face swap with an optional private Tailscale HTTP/WebSocket controller for the Windows app.
"""ENDMARKDOWN

# %% [markdown] cell=1 id=setup-heading
"""MARKDOWN
## 1. Clone repository and install dependencies
"""ENDMARKDOWN

# %% [code] cell=2 id=setup
"""CELL: Clone and install"""
# @title Clone and install
import os
import subprocess
import sys
from pathlib import Path

# Clone the repository
REPO_URL = "https://github.com/djebaz/AiSwap.git"
WORK_DIR = Path("/content/Deep-Live-Cam-Remote")

if WORK_DIR.exists():
    print(f"Removing existing directory: {WORK_DIR}")
    import shutil
    shutil.rmtree(WORK_DIR)

# Clone into temp location then move the subdirectory
TEMP_CLONE = Path("/content/AiSwap_temp")
if TEMP_CLONE.exists():
    import shutil
    shutil.rmtree(TEMP_CLONE)

print(f"Cloning {REPO_URL}...")
subprocess.run(["git", "clone", "--depth=1", "--branch=main", REPO_URL, str(TEMP_CLONE)], check=True)

# Move the Deep-Live-Cam-Remote subdirectory
import shutil
source_dir = TEMP_CLONE / "projects" / "Deep-Live-Cam-Remote"
shutil.move(str(source_dir), str(WORK_DIR))
shutil.rmtree(TEMP_CLONE)

print(f"Repository cloned to: {WORK_DIR}")

# Install dependencies
print("Installing Python dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy<2",
    "opencv-python==4.10.0.84",
    "insightface==0.7.3",
    "onnx==1.18.0",
    "onnxruntime-gpu==1.23.2",
    "scikit-learn",
    "tqdm",
    "pillow",
    "psutil",
    "protobuf==4.25.1",
    "PySide6>=6.7,<7",
    "cv2_enumerate_cameras==1.1.15",
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
    "websockets>=12,<16"
], check=True)

# Download the face swapper model
import urllib.request
MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
MODEL_PATH = WORK_DIR / "models" / "inswapper_128.onnx"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1024 * 1024:
    MODEL_PATH.unlink(missing_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".onnx.part")
    temporary_model.unlink(missing_ok=True)
    print("Downloading face swapper model to", MODEL_PATH)
    urllib.request.urlretrieve(MODEL_URL, temporary_model)
    if temporary_model.stat().st_size < 1024 * 1024:
        temporary_model.unlink(missing_ok=True)
        raise RuntimeError("Downloaded inswapper_128.onnx is incomplete")
    temporary_model.replace(MODEL_PATH)

print(f"Face swapper model ready: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1048576:.1f} MB)")

# Clean up any cached modules from previous runs
for module_name in list(sys.modules):
    if module_name == "colab_batch" or module_name == "modules" or module_name.startswith("modules."):
        del sys.modules[module_name]

# Set working directory and path
os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

# Show GPU info
subprocess.run(["nvidia-smi"], check=False)
print("Runtime ready:", WORK_DIR)

# %% [markdown] cell=4 id=config-heading
"""MARKDOWN
## 2. Configure Colab paths and processing options
"""ENDMARKDOWN

# %% [code] cell=5 id=config
"""CELL: Batch configuration"""
# @title Batch configuration
DRIVE_ROOT = "/content/drive/MyDrive/DeepLiveCamRemote"
SOURCE_FACE = DRIVE_ROOT + "/source/source.png"
INPUT_DIR = DRIVE_ROOT + "/videos"
PHOTO_INPUT_DIR = DRIVE_ROOT + "/photos"
OUTPUT_DIR = DRIVE_ROOT + "/outputs/videos"
PHOTO_OUTPUT_DIR = DRIVE_ROOT + "/outputs/photos"
ZIP_PATH = DRIVE_ROOT + "/outputs/face_swapped_outputs.zip"
SS = 0.0
DURATION = None  # None processes the remainder
MAX_FPS = 30.0
MAX_WIDTH = 420
MANY_FACES = False
OPACITY = 1.0
SHARPNESS = 0.0
MOUTH_MASK_SIZE = 0.0
POISSON_BLEND = False
COLOR_CORRECTION = False
INTERPOLATION_WEIGHT = 0.0
ENHANCER = "none"  # none, gfpgan, gpen256, gpen512
MAPPING_JSON = None  # e.g. "/content/mapping/face_mapping.json"

# %% [markdown] cell=6 id=mapping-heading
"""MARKDOWN
## 3. Optional: scan identities and edit mapping JSON
Run this before processing only when different target identities need different source faces. Set each generated `source_path`, then set `MAPPING_JSON` above.
"""ENDMARKDOWN

# %% [code] cell=7 id=mapping
"""CELL: Scan identity gallery (optional)"""
# @title Scan identity gallery (optional)
from colab_batch import main
MAPPING_DIR = "/content/mapping"
main(["scan", "--input-dir", INPUT_DIR, "--mapping-dir", MAPPING_DIR])

# %% [markdown] cell=8 id=process-heading
"""MARKDOWN
## 4. Process folder and create ZIP
"""ENDMARKDOWN

# %% [code] cell=9 id=process
"""CELL: Run batch processor"""
# @title Run batch processor
from colab_batch import main
args = ["process", "--input-dir", INPUT_DIR, "--output-dir", OUTPUT_DIR, "--zip-output", ZIP_PATH, "--ss", str(SS), "--max-fps", str(MAX_FPS), "--max-width", str(MAX_WIDTH), "--opacity", str(OPACITY), "--sharpness", str(SHARPNESS), "--mouth-mask-size", str(MOUTH_MASK_SIZE), "--interpolation-weight", str(INTERPOLATION_WEIGHT), "--enhancer", ENHANCER]
if SOURCE_FACE: args += ["--source-face", SOURCE_FACE]
if DURATION is not None: args += ["--duration", str(DURATION)]
if MAPPING_JSON: args += ["--map-config", MAPPING_JSON]
if MANY_FACES: args += ["--many-faces"]
if POISSON_BLEND: args += ["--poisson-blend"]
if COLOR_CORRECTION: args += ["--color-correction"]
exit_code = main(args)
print("Batch exit code:", exit_code)

# %% [markdown] cell=10 id=download-heading
"""MARKDOWN
## 5. Download ZIP
"""ENDMARKDOWN

# %% [code] cell=11 id=download
"""CELL: Download result archive"""
# @title Download result archive
from google.colab import files
files.download(ZIP_PATH)

# %% [markdown] cell=12 id=api-heading
"""MARKDOWN
## 6. Optional: start private Windows app API
Run this after connecting Colab to Tailscale. The Windows app connects to `http://TAILSCALE_IP:7860`.
"""ENDMARKDOWN

# %% [code] cell=13 id=start-api
"""CELL: Start private API server"""
# @title Start private API server
from colab_api import ensure_drive_layout, main as api_main
ensure_drive_layout()
api_main(["--host", "0.0.0.0", "--port", "7860"])

# %% [markdown] cell=14 id=photos-heading
"""MARKDOWN
## 7. Optional: run photo batch directly in Colab
"""ENDMARKDOWN

# %% [code] cell=15 id=photos
"""CELL: Run photo batch processor"""
# @title Run photo batch processor
from colab_batch import main
photo_args = ["photos", "--input-dir", PHOTO_INPUT_DIR, "--output-dir", PHOTO_OUTPUT_DIR, "--source-face", SOURCE_FACE, "--opacity", str(OPACITY), "--sharpness", str(SHARPNESS), "--mouth-mask-size", str(MOUTH_MASK_SIZE), "--interpolation-weight", str(INTERPOLATION_WEIGHT), "--enhancer", ENHANCER]
if MANY_FACES: photo_args += ["--many-faces"]
if POISSON_BLEND: photo_args += ["--poisson-blend"]
if COLOR_CORRECTION: photo_args += ["--color-correction"]
exit_code = main(photo_args)
print("Photo batch exit code:", exit_code)
