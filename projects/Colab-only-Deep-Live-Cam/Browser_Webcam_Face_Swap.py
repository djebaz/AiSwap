# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJwcm92ZW5hbmNlIjpbXX0sImtlcm5lbHNwZWMiOnsibmFtZSI6InB5dGhvbjMiLCJkaXNwbGF5X25hbWUiOiJQeXRob24gMyJ9LCJsYW5ndWFnZV9pbmZvIjp7Im5hbWUiOiJweXRob24ifSwibmJmb3JtYXQiOjQsIm5iZm9ybWF0X21pbm9yIjowfQ==

# %% [markdown] cell=0 id=overview
"""MARKDOWN
# Browser-Based Real-Time Face Swap with Colab GPU

This notebook captures your webcam directly in the browser, processes frames on Colab's GPU, and displays the face-swapped result in real-time.

**No Windows client needed!** Everything runs in this notebook.

## Steps:
1. Select a GPU runtime (T4 recommended)
2. Install dependencies
3. Upload a source face image
4. Start webcam capture and processing
"""ENDMARKDOWN

# %% [markdown] cell=1 id=install
"""MARKDOWN
## 1. Install Dependencies
"""ENDMARKDOWN

# %% [code] cell=2 id=install-deps
"""CELL: %%capture"""
%%capture
!pip install huggingface_hub
!pip install onnxruntime-gpu==1.20.1 #--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
!pip install insightface #==0.7.3
!pip install fastrtc
# %% [markdown] cell=3 id=verify
"""MARKDOWN
## 2. Verify GPU
"""ENDMARKDOWN

# %% [code] cell=4 id=verify-gpu
"""CELL: import torch"""
import torch
import onnxruntime as ort

print("CUDA available:", torch.cuda.is_available())
print("ONNX Runtime providers:", ort.get_available_providers())

assert "CUDAExecutionProvider" in ort.get_available_providers(), "GPU not available!"

# %% [markdown] cell=5 id=models
"""MARKDOWN
## 3. Download Face Swap Model
"""ENDMARKDOWN

# %% [code] cell=6 id=download-models
"""CELL: import os"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from google.colab import userdata

MODEL_DIR = Path("/content/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = "ninjawick/webui-faceswap-unlocked"
HF_FILENAME = "inswapper_128.onnx"
SWAPPER_PATH = MODEL_DIR / HF_FILENAME

if not SWAPPER_PATH.exists():
    print(f"Downloading {HF_FILENAME} to {MODEL_DIR}...")
    try:
        # Get token if available
        try:
            hf_token = userdata.get('HF_TOKEN')
        except:
            hf_token = None

        # Download directly to the local directory
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=str(MODEL_DIR),
            token=hf_token
        )

        print(f"Successfully downloaded to: {downloaded_path}")
        print(f"File size: {os.path.getsize(SWAPPER_PATH) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("If this is a private repo, ensure your HF_TOKEN is set in Colab secrets.")
else:
    print(f"Model already exists at: {SWAPPER_PATH}")
# %% [markdown] cell=7 id=init
"""MARKDOWN
## 4. Initialize Face Processing
"""ENDMARKDOWN

# %% [code] cell=8 id=init-face-processing
"""CELL: import cv2"""
import cv2
import insightface
import numpy as np

ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize face analyzer
print("Initializing face analyzer...")
FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=ORT_PROVIDERS)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

# Initialize face swapper
print("Initializing face swapper...")
FACE_SWAPPER = insightface.model_zoo.get_model(str(SWAPPER_PATH), providers=ORT_PROVIDERS)

print("✓ GPU face processing initialized")

def get_one_face(frame):
    """Detect the first face in the frame."""
    faces = FACE_ANALYSER.get(frame)
    if not faces:
        return None
    return min(faces, key=lambda face: face.bbox[0])

def swap_face(source_face, target_frame):
    """Swap source face into target frame."""
    target_face = get_one_face(target_frame)
    if target_face is None:
        return target_frame  # No face detected, return original
    
    result = FACE_SWAPPER.get(target_frame, target_face, source_face, paste_back=True)
    return result

# %% [markdown] cell=9 id=upload
"""MARKDOWN
## 5. Upload Source Face Image

Upload an image containing the face you want to swap onto your webcam.
"""ENDMARKDOWN

# %% [code] cell=10 id=upload-source
"""CELL: from google.colab import files"""
from google.colab import files
from IPython.display import Image, display
import io

print("Upload a source face image (JPG, PNG):")
uploaded = files.upload()

# Get the uploaded file
source_filename = list(uploaded.keys())[0]
source_bytes = uploaded[source_filename]

# Decode image
source_array = np.frombuffer(source_bytes, np.uint8)
SOURCE_IMAGE = cv2.imdecode(source_array, cv2.IMREAD_COLOR)

print(f"Source image loaded: {SOURCE_IMAGE.shape}")
display(Image(data=source_bytes, width=300))

# Extract source face
print("\nDetecting source face...")
SOURCE_FACE = get_one_face(SOURCE_IMAGE)
if SOURCE_FACE is None:
    raise ValueError("No face detected in source image!")
print("✓ Source face detected and cached")

# %% [markdown] cell=11 id=webcam
"""MARKDOWN
## 6. Start Real-Time Face Swap

This will:
1. Capture your webcam in the browser
2. Send frames to Colab GPU for processing
3. Display the face-swapped result in real-time

**Click "Allow" when prompted for webcam access.**

Press **Stop** button to end the stream.
"""ENDMARKDOWN

# %% [code] cell=12 id=webcam-stream
"""CELL: from IPython.display import display, Javascript, HTML"""
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL.Image
import io
import time

def webcam_to_numpy(quality=0.8, size=(640, 480)):
    """Capture a frame from webcam and return as numpy array."""
    js = Javascript('''
    async function captureFrame(quality, width, height) {
      const video = document.createElement('video');
      const stream = await navigator.mediaDevices.getUserMedia({video: {width, height}});
      
      video.srcObject = stream;
      await video.play();
      
      // Wait for video to be ready
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      
      stream.getTracks().forEach(track => track.stop());
      
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    
    data = eval_js(f'captureFrame({quality}, {size[0]}, {size[1]})')
    binary = b64decode(data.split(',')[1])
    
    img = PIL.Image.open(io.BytesIO(binary))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

print("Starting webcam... (this may take a moment)")
print("="*50)

# Test capture
test_frame = webcam_to_numpy(quality=0.8, size=(640, 480))
print(f"✓ Webcam ready! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
print("\nProcessing first frame...")

# Process and display
result = swap_face(SOURCE_FACE, test_frame)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
result_pil = PIL.Image.fromarray(result_rgb)

display(result_pil)
print("\n✓ Face swap working!")
print("\nRun the next cell for continuous streaming.")

# %% [markdown] cell=13 id=stream
"""MARKDOWN
## 7. Continuous Stream (Run this for live video)

**Note:** Due to Colab limitations, this captures frames repeatedly rather than true video streaming. Expect 2-5 FPS.

Press **Interrupt** (⏹️) to stop.
"""ENDMARKDOWN

# %% [code] cell=14 id=continuous-stream
"""CELL: from fastrtc import ("""
from fastrtc import (
    Stream,
    VideoStreamHandler,
    get_cloudflare_turn_credentials,
    get_cloudflare_turn_credentials_async,
)
import cv2

RTC_HF_TOKEN = userdata.get("HF_TOKEN")
if not RTC_HF_TOKEN:
    raise ValueError("Add HF_TOKEN to Colab Secrets and enable notebook access.")

async def get_rtc_credentials():
    return await get_cloudflare_turn_credentials_async(
        hf_token=RTC_HF_TOKEN,
        ttl=3600,
    )

def process_stream_frame(rgb_frame):
    """Apply face swapping to one FastRTC RGB webcam frame."""
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    result = swap_face(SOURCE_FACE, bgr_frame)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

stream = Stream(
    handler=VideoStreamHandler(process_stream_frame, skip_frames=True),
    modality="video",
    mode="send-receive",
    rtc_configuration=get_rtc_credentials,
    server_rtc_configuration=get_cloudflare_turn_credentials(
        hf_token=RTC_HF_TOKEN,
        ttl=3600,
    ),
)

stream.ui.launch(share=True)
# %% [markdown] cell=15 id=notes
"""MARKDOWN
## Notes

- **Performance**: Browser → Colab → Browser adds latency. Expect 2-5 FPS.
- **Privacy**: Your webcam data stays in your browser and Colab session. Nothing is stored.
- **GPU**: Uses Colab's T4 GPU for real-time face swapping.
- **Limitations**: This won't work in Zoom/Teams directly - it's just for preview.

For production use with virtual cameras, use the Windows client version instead.
"""ENDMARKDOWN
