# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJwcm92ZW5hbmNlIjpbXSwiY29sbGFwc2VkX3NlY3Rpb25zIjpbImJhdGNoLXZpZGVvLWhlYWRpbmciXX0sImtlcm5lbHNwZWMiOnsibGFuZ3VhZ2UiOiJweXRob24iLCJuYW1lIjoicHl0aG9uMyIsImRpc3BsYXlfbmFtZSI6IlB5dGhvbiAzIn0sImxhbmd1YWdlX2luZm8iOnsibmFtZSI6InB5dGhvbiJ9LCJuYmZvcm1hdCI6NCwibmJmb3JtYXRfbWlub3IiOjB9

# %% [markdown] cell=0
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

# %% [markdown] cell=1
"""MARKDOWN
## 1. Install Dependencies
"""ENDMARKDOWN

# %% [code] cell=2
"""CELL: Install dependencies"""
# @title Install dependencies
%pip install "numpy<2" insightface==0.7.3 --no-deps
%pip uninstall -y onnxruntime onnxruntime-gpu
!pip install huggingface_hub
!pip install onnxruntime-gpu==1.20.1 #--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

!pip install onnx==1.16.0

# %% [markdown] cell=3
"""MARKDOWN
## 2. Verify GPU
"""ENDMARKDOWN

# %% [code] cell=4
"""CELL: import torch"""
import torch
import onnxruntime as ort

print("CUDA available:", torch.cuda.is_available())
print("ONNX Runtime providers:", ort.get_available_providers())

assert "CUDAExecutionProvider" in ort.get_available_providers(), "GPU not available!"

# %% [markdown] cell=5
"""MARKDOWN
## 3. Download Face Swap Model
"""ENDMARKDOWN

# %% [code] cell=6
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

# %% [markdown] cell=7
"""MARKDOWN
## 4. Initialize Face Processing
"""ENDMARKDOWN

# %% [code] cell=8
"""CELL: import cv2"""
import cv2
import insightface
import numpy as np

ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Initialize face analyzer with a smaller detection size for faster inference
print("Initializing face analyzer...")
FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=ORT_PROVIDERS)
# Reduced det_size from 640 to 320 to speed up detection
FACE_ANALYSER.prepare(ctx_id=0, det_size=(320, 320))

# Initialize face swapper
print("Initializing face swapper...")
FACE_SWAPPER = insightface.model_zoo.get_model(str(SWAPPER_PATH), providers=ORT_PROVIDERS)

print("✓ GPU face processing initialized with optimized detection size")

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
        return target_frame

    result = FACE_SWAPPER.get(target_frame, target_face, source_face, paste_back=True)
    return result

# %% [markdown] cell=9
"""MARKDOWN
## 5. Upload Source Face Image

Upload an image containing the face you want to swap onto your webcam.
"""ENDMARKDOWN

# %% [code] cell=10
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

# %% [markdown] cell=11
"""MARKDOWN
## 6. Start Real-Time Face Swap

This will:
1. Capture your webcam in the browser
2. Send frames to Colab GPU for processing
3. Display the face-swapped result in real-time

**Click "Allow" when prompted for webcam access.**

Press **Stop** button to end the stream.
"""ENDMARKDOWN

# %% [code] cell=12
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

# %% [markdown] cell=13
"""MARKDOWN
## 7. Continuous Stream (Run this for live video)

**Note:** Due to Colab limitations, this captures frames repeatedly rather than true video streaming. Expect 2-5 FPS.

Press **Interrupt** (⏹️) to stop.
"""ENDMARKDOWN

# %% [code] cell=14
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
    server_rtc_configuration=get_server_rtc_credentials(),
)

stream.ui.launch(share=True)

# %% [markdown] cell=15
"""MARKDOWN
### 8. Improved Smoothing (Optional)
To make the video appear smoother at lower frame rates, we can use a weighted average of the current and previous frame. This creates a motion-blur effect that masks the stuttering.
"""ENDMARKDOWN

# %% [code] cell=16
"""CELL: import numpy as np"""
import numpy as np

# Global variable to store the previous frame for smoothing
prev_frame = None
SMOOTHING_FACTOR = 0.0  # Adjust between 0.0 (no smoothing) and 0.9 (heavy blur)

def process_smoothed_frame(rgb_frame):
    global prev_frame

    # 1. Process current frame (Face Swap)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    current_result = swap_face(SOURCE_FACE, bgr_frame)

    # 2. Apply smoothing if a previous frame exists
    if prev_frame is not None:
        # Blend: (1 - factor) * current + factor * previous
        smoothed = cv2.addWeighted(current_result, 1 - SMOOTHING_FACTOR, prev_frame, SMOOTHING_FACTOR, 0)
        prev_frame = smoothed
        return cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)

    prev_frame = current_result
    return cv2.cvtColor(current_result, cv2.COLOR_BGR2RGB)

# Launching a new stream with smoothing enabled
smooth_stream = Stream(
    handler=VideoStreamHandler(process_smoothed_frame, skip_frames=True),
    modality="video",
    mode="send-receive",
    rtc_configuration=get_rtc_credentials,
    server_rtc_configuration=get_server_rtc_credentials(),
)

smooth_stream.ui.launch(share=True)

# %% [markdown] cell=17
"""MARKDOWN
### 9. Fallback: Gradio HTTP Streaming
If FastRTC's WebRTC connection continues to fail due to network/DNS issues, use this standard Gradio implementation. It uses a regular HTTP/Websocket stream which is more robust in restricted environments.
"""ENDMARKDOWN

# %% [code] cell=18
"""CELL: import gradio as gr"""
import gradio as gr
import cv2
import numpy as np

# State to keep track of previous frames for smoothing
prev_frame_gradio = None
SMOOTHING_FACTOR = 0.0  # Adjust 0.0 to 0.9 for more/less smoothing

def gradio_stream_handler(frame):
    global prev_frame_gradio
    if frame is None:
        return None

    # 1. Convert RGB to BGR for insightface processing
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 2. Apply Face Swap
    current_result = swap_face(SOURCE_FACE, bgr_frame)

    # 3. Apply Smoothing logic
    if prev_frame_gradio is not None:
        # Ensure dimensions match for blending
        if current_result.shape == prev_frame_gradio.shape:
            current_result = cv2.addWeighted(
                current_result, 1 - SMOOTHING_FACTOR,
                prev_frame_gradio, SMOOTHING_FACTOR, 0
            )

    prev_frame_gradio = current_result.copy()

    # 4. Convert back to RGB for Gradio display
    return cv2.cvtColor(current_result, cv2.COLOR_BGR2RGB)

# CSS to hide the input container and make the output very large
css = """
#webcam_input { display: none !important; }
#output_video { height: 80vh !important; }
.gradio-container { max-width: 100% !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Real-Time Face Swap (Output Only)")

    # We keep the component so the stream works, but hide it via CSS
    webcam_input = gr.Image(
        sources=['webcam'],
        streaming=True,
        elem_id="webcam_input"
    )

    output_video = gr.Image(
        label="Face Swap Result",
        elem_id="output_video"
    )

    # Trigger processing on every new frame from the stream
    webcam_input.stream(
        fn=gradio_stream_handler,
        inputs=webcam_input,
        outputs=output_video,
        time_limit=600,
        stream_every=0.02
    )

demo.launch(share=True, inline=False)

# %% [code] cell=19
"""CELL: !pip install -q gradio"""
!pip install -q gradio

import gradio as gr
import cv2

PROCESS_WIDTH = 480
PROCESS_HEIGHT = 360

def gradio_stream_handler(frame):
    if frame is None:
        return None

    frame = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = swap_face(SOURCE_FACE, bgr)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


css = """
.gradio-container {
    max-width: 1400px !important;
}

#swap-output {
    width: 100% !important;
}

#swap-output img {
    object-fit: contain !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Real-Time Face Swap")

    webcam_input = gr.Image(
        sources="webcam",
        type="numpy",
        streaming=True,
        height=360,
        label="Webcam",
    )

    output_image = gr.Image(
        type="numpy",
        format="jpeg",
        streaming=True,
        elem_id="swap-output",
        label="Face Swap Output",
    )

    webcam_input.stream(
        fn=gradio_stream_handler,
        inputs=webcam_input,
        outputs=output_image,
        stream_every=0.05,
        time_limit=300,
        concurrency_limit=1,
    )

demo.launch(share=True, inline=False)

# %% [code] cell=20
"""CELL: import gradio as gr"""
import gradio as gr
import cv2
import collections
import time

# SETTINGS FOR PARALLEL GPU SATURATION
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240
STREAM_EVERY = 0.01
MAX_CONCURRENT_FRAMES = 8
BUFFER_TARGET_SIZE = 15 # Reduced slightly for faster startup
frame_buffer = collections.deque(maxlen=100)
last_valid_frame = None

processing_times = collections.deque(maxlen=100)

def gradio_stream_handler(frame):
    global frame_buffer, last_valid_frame
    if frame is None:
        return last_valid_frame

    start_time = time.time()

    # 1. GPU Processing
    resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
    bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    swapped = swap_face(SOURCE_FACE, bgr_frame)
    rgb_out = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)

    # 2. Profiling
    proc_duration = (time.time() - start_time) * 1000
    processing_times.append(proc_duration)

    if len(processing_times) % 20 == 0:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"[Profile] Avg GPU Proc: {avg_time:.2f}ms | Buffer: {len(frame_buffer)}/{BUFFER_TARGET_SIZE}")

    # 3. Buffering
    frame_buffer.append(rgb_out)

    if len(frame_buffer) >= BUFFER_TARGET_SIZE:
        last_valid_frame = frame_buffer.popleft()
        return last_valid_frame

    return last_valid_frame

css = """
.gradio-container { max-width: 100% !important; margin: 0 !important; }
#webcam-input { max-width: 300px !important; margin: 0 auto 10px auto !important; }
#swap-output { width: 100% !important; height: 85vh !important; }
#swap-output img { object-fit: contain !important; height: 100% !important; width: 100% !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# High-FPS Parallelized Face Swap (JPEG Optimized)")

    with gr.Column():
        webcam_input = gr.Image(sources="webcam", type="numpy", streaming=True, elem_id="webcam-input")
        # Switched format to jpeg for better compatibility
        output_image = gr.Image(type="numpy", format="jpeg", streaming=True, elem_id="swap-output")

    webcam_input.stream(
        fn=gradio_stream_handler,
        inputs=webcam_input,
        outputs=output_image,
        stream_every=STREAM_EVERY,
        time_limit=1200,
        concurrency_limit=MAX_CONCURRENT_FRAMES,
    )

demo.queue(default_concurrency_limit=MAX_CONCURRENT_FRAMES)
demo.launch(share=True, inline=False, debug=True)

# %% [code] cell=21
"""CELL: import gradio as gr"""
import gradio as gr
import cv2
import collections
import time
import numpy as np

# --- HIGH FLUIDITY STREAM CONFIGURATION ---
PROCESS_WIDTH = 480
PROCESS_HEIGHT = 360
STREAM_EVERY = 0.05  # 50ms (Targets ~20 FPS for a fluid video feel)
MAX_CONCURRENT_FRAMES = 2
BUFFER_TARGET_SIZE = 4

# Load specific source image selected by user
SELECTED_SOURCE_PATH = '/content/Capture d’écran 2026-06-21 025249.jpg'
source_img = cv2.imread(SELECTED_SOURCE_PATH)
SOURCE_FACE = get_one_face(source_img)

frame_buffer = collections.deque(maxlen=20)
last_valid_frame = None

def optimized_stream_handler(frame):
    global frame_buffer, last_valid_frame

    if frame is None:
        return last_valid_frame

    try:
        # 1. Processing (keeping resolution balanced for speed)
        resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_LINEAR)
        bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        swapped = swap_face(SOURCE_FACE, bgr_frame)
        rgb_out = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)

        # 2. Add to buffer
        frame_buffer.append(rgb_out)

        # 3. Stream delivery
        if len(frame_buffer) >= BUFFER_TARGET_SIZE:
            last_valid_frame = frame_buffer.popleft()
            return last_valid_frame

    except Exception as e:
        pass # Silent fail to keep the video moving

    return last_valid_frame

css = """
.gradio-container { max-width: 100% !important; margin: 0 !important; }
#webcam-input { max-width: 250px !important; margin: 0 auto 10px auto !important; opacity: 0.5; }
#swap-output { width: 100% !important; height: 85vh !important; }
#swap-output img { object-fit: contain !important; height: 100% !important; width: 100% !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("### Live Video Stream: [High Fluidity] | [20 FPS Target] | [Buffer Fallback Enabled]")

    with gr.Column():
        webcam_input = gr.Image(sources="webcam", type="numpy", streaming=True, elem_id="webcam-input")
        output_image = gr.Image(type="numpy", format="jpeg", streaming=True, elem_id="swap-output", label="Face-Swapped Stream")

    webcam_input.stream(
        fn=optimized_stream_handler,
        inputs=webcam_input,
        outputs=output_image,
        stream_every=STREAM_EVERY,
        time_limit=1800,
        concurrency_limit=MAX_CONCURRENT_FRAMES,
    )

demo.queue(default_concurrency_limit=MAX_CONCURRENT_FRAMES)
demo.launch(share=True, inline=False, debug=False)

# %% [markdown] cell=22
"""MARKDOWN
### 10. True Video Streaming with FFmpeg
This method uses FFmpeg to wrap the processed frames into a streamable format, which is then played back in an HTML5 Video Player.
"""ENDMARKDOWN

# %% [code] cell=23
"""CELL: import os"""
import os
import subprocess
import threading
import base64
import cv2
import PIL.Image
import io
from google.colab import output
from IPython.display import HTML, display
from base64 import b64decode

# --- CONFIG ---
WIDTH, HEIGHT = 640, 480
FPS = 10
OUTPUT_PATH = '/content/output_face_swap.mp4'

# FFmpeg command using 'tee' to save to disk AND pipe to the browser player
# Modified movflags for better compatibility and removed zerolatency tune
save_ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
    "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", # Changed preset from ultrafast to medium
    "-f", "tee", "-map", "0:v", f"{OUTPUT_PATH}|[f=mp4:movflags=+faststart]-" # Changed movflags to +faststart
]

class VideoStreamServer:
    def __init__(self, cmd):
        self.process = None
        self.chunk_count = 0
        self.cmd = cmd

    def start(self):
        self.process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._relay_stdout, daemon=True).start()

    def _relay_stdout(self):
        while self.process:
            chunk = self.process.stdout.read(8192)
            if not chunk: break
            self.chunk_count += 1
            encoded_chunk = base64.b64encode(chunk).decode("utf-8")
            output.eval_js(f'window.receiveChunk("{encoded_chunk}", {self.chunk_count})')

    def push_frame(self, frame):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except:
                pass

    def stop(self):
        if self.process:
            try:
                self.process.stdin.close() # Close stdin to signal EOF to ffmpeg
                self.process.wait(timeout=10) # Wait for ffmpeg to finish writing output
                if self.process.poll() is None:
                    self.process.terminate()
                print("FFmpeg process stopped successfully.")
            except subprocess.TimeoutExpired:
                print("FFmpeg process did not terminate gracefully, killing it.")
                self.process.kill()
            except Exception as e:
                print(f"Error stopping FFmpeg process: {e}")
            finally:
                self.process = None

# Re-initialize the server globally so the other cell can use it
stream_server = VideoStreamServer(save_ffmpeg_cmd)
stream_server.start()

def process_and_stream(image_b64):
    try:
        binary = b64decode(image_b64.split(",")[1])
        img = PIL.Image.open(io.BytesIO(binary))
        frame = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (WIDTH, HEIGHT))
        swapped = swap_face(SOURCE_FACE, frame)
        stream_server.push_frame(swapped)
    except:
        pass

output.register_callback("notebook.process_and_stream", process_and_stream)

print(f"✓ Recording Initialized!")
print(f"Video will be saved to: {OUTPUT_PATH}")
display(HTML("<div style='color: #00ff00; font-weight: bold; border: 1px solid #00ff00; padding: 10px;'>Recording is now ACTIVE. Please re-run the 'True Video Streaming' cell (737db946) to begin.</div>"))

# %% [code] cell=24
"""CELL: import cv2"""
import cv2
import numpy as np
import base64
from google.colab import output
from google.colab.output import eval_js
from base64 import b64decode
import PIL.Image
import io
from IPython.display import HTML, display

# --- CONFIG ---
# Dimensions and FPS are inherited from the recording initialization cell (16cd161a)

# Update Source Face from selected file (Handling the specific filename with spaces)
try:
    selected_path = "/content/image_001_proc.jpg"
    source_img = cv2.imread(selected_path)
    if source_img is None:
        raise FileNotFoundError(f"File not found at {selected_path}")
    SOURCE_FACE = get_one_face(source_img)
    print(f"✓ Success: Loaded source face from {selected_path}")
except Exception as e:
    print(f"☁ Fallback: Using previous face due to error: {e}")

def process_and_stream(image_b64):
    try:
        binary = b64decode(image_b64.split(",")[1])
        img = PIL.Image.open(io.BytesIO(binary))
        frame = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (WIDTH, HEIGHT))

        # Process frame with face swap
        swapped = swap_face(SOURCE_FACE, frame)

        # Push to the stream_server (which handles UI playback and MP4 recording)
        if 'stream_server' in globals():
            stream_server.push_frame(swapped)
    except Exception as e:
        pass

output.register_callback("notebook.process_and_stream", process_and_stream)

html_code = f'''
<div style="text-align:center; background:#1e1e1e; padding: 20px; border-radius: 10px;">
    <video id="v" autoplay playsinline muted style="width:80%; max-width: 640px; border: 1px solid #444;"></video>
    <canvas id="c" style="display:none;"></canvas>
    <div style="color: #00ff00; font-family: monospace; margin-top: 10px; font-size: 14px;">
      Status: <span id="st">Waiting...</span> | Chunks: <span id="cc">0</span> | <span style="color:red">● RECORDING TO DISK</span>
    </div>
</div>
<script>
    const v = document.getElementById("v");
    const st = document.getElementById("st");
    const cc = document.getElementById("cc");
    const ms = new MediaSource();
    v.src = URL.createObjectURL(ms);
    let sb; let queue = [];

    ms.onsourceopen = () => {{
        sb = ms.addSourceBuffer('video/mp4; codecs="avc1.42E01E"');
        sb.mode = 'sequence';
        window.receiveChunk = (base64, count) => {{
            cc.innerText = count;
            st.innerText = "Streaming";
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            if (sb.updating || queue.length > 0) {{ queue.push(bytes); }}
            else {{ try {{ sb.appendBuffer(bytes); }} catch(e) {{}} }}
        }};
        sb.addEventListener('updateend', () => {{
            if (queue.length > 0 && !sb.updating) {{ sb.appendBuffer(queue.shift()); }}
        }});
    }};

    async function start() {{
        try {{
            const stream = await navigator.mediaDevices.getUserMedia({{video: {{width: {WIDTH}, height: {HEIGHT}}}}});
            const cam = document.createElement("video");
            cam.srcObject = stream; await cam.play();
            const canvas = document.getElementById("c");
            canvas.width = {WIDTH}; canvas.height = {HEIGHT};
            const ctx = canvas.getContext("2d");
            setInterval(() => {{
                ctx.drawImage(cam, 0, 0, {WIDTH}, {HEIGHT});
                google.colab.kernel.invokeFunction("notebook.process_and_stream", [canvas.toDataURL("image/jpeg", 0.6)], {{}});
            }}, 1000 / {FPS});
        }} catch (e) {{ st.innerText = "Error: " + e.message; }}
    }}
    start();
</script>
'''
display(HTML(html_code))

# %% [markdown] cell=25
"""MARKDOWN
### Don't forget to stop the recording!

Once you are done with the live stream (by pressing the 'Interrupt' button on the cell above), **remember to run the 'Stop Recording and Finalize Video' cell** (the one with `stop_recording()`) to ensure your `output_face_swap.mp4` file is saved correctly without corruption.
"""ENDMARKDOWN

# %% [code] cell=26
"""CELL: def stop_recording():"""
def stop_recording():
    if 'stream_server' in globals() and stream_server.process:
        print("Stopping FFmpeg recording and finalizing video...")
        stream_server.stop()
        print(f"✓ Recording stopped. Video saved to: {OUTPUT_PATH}")
    else:
        print("Recording server not active or already stopped.")

# You can call stop_recording() directly to stop it.
# For example, to stop immediately after streaming ends:
stop_recording()

# %% [markdown] cell=27
"""MARKDOWN
## Notes

- **Performance**: Browser → Colab → Browser adds latency. Expect 2-5 FPS.
- **Privacy**: Your webcam data stays in your browser and Colab session. Nothing is stored.
- **GPU**: Uses Colab's T4 GPU for real-time face swapping.
- **Limitations**: This won't work in Zoom/Teams directly - it's just for preview.

For production use with virtual cameras, use the Windows client version instead.

## 12. Repair MP4 File (If still corrupted)

If the generated `output_face_swap.mp4` file is still corrupted or unplayable, this step attempts to repair it by re-encoding the video using FFmpeg. This can often fix issues with file headers or fragmented streams.
"""ENDMARKDOWN

# %% [code] cell=28
"""CELL: import subprocess"""
import subprocess
import os

INPUT_FILE = '/content/output_face_swap.mp4'
OUTPUT_FILE = '/content/output_face_swap_repaired.mp4'

if os.path.exists(INPUT_FILE):
    print(f"Attempting to repair and re-encode {INPUT_FILE}...")
    # Removed -c:v copy and -c:a copy to force re-encoding for full repair
    repair_cmd = [
        "ffmpeg",
        "-y", # Overwrite output files without asking
        "-i", INPUT_FILE, # Input file
        "-c:v", "libx264", # Force re-encode video with H.264
        "-c:a", "aac",    # Force re-encode audio with AAC
        "-movflags", "faststart", # Optimize for streaming
        OUTPUT_FILE # Output file
    ]

    try:
        subprocess.run(repair_cmd, check=True, capture_output=True)
        print(f"✓ Repair attempt complete. Repaired video saved to: {OUTPUT_FILE}")
        print("You can now download and check 'output_face_swap_repaired.mp4'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during repair: {e.stderr.decode()}")
        print(f"Repair failed for {INPUT_FILE}.")
else:
    print(f"Input file {INPUT_FILE} not found. Please ensure the video was generated first.")

# %% [markdown] cell=29
"""MARKDOWN
### 11. Save Output to Disk
Run the cell below to modify the streaming logic so that it also records the video to a file named `output_face_swap.mp4`.

### Important: Stop Recording to Finalize the File

After you've finished streaming (by interrupting the 'True Video Streaming' cell), **you MUST run the Python cell above this markdown cell** to properly stop the FFmpeg process and finalize the `output_face_swap.mp4` file. If you don't, the video file will likely be corrupted or unplayable (missing the `moov` atom).
"""ENDMARKDOWN

# %% [markdown] cell=30 id=batch-video-heading
"""MARKDOWN
## 13. Batch-process video folders

Run one of the following cells after the face-analysis and `swap_face()` setup cells. Both use paths directly in `/content`; no upload widget is required.
"""ENDMARKDOWN

# %% [code] cell=31 id=batch-video-cv2
"""CELL: Batch video folder (OpenCV + FFmpeg)"""
# @title Batch video folder (OpenCV + FFmpeg)
import cv2
import subprocess
import shutil
from pathlib import Path
from tqdm.auto import tqdm

# Assumes get_one_face() and swap_face() were initialized by earlier notebook cells.

SOURCE_FACE_PATH = Path("/content/vi_0003_portrait.png")
INPUT_VIDEO_DIR = Path("/content/input_videos")
OUTPUT_VIDEO_DIR = Path("/content/output_videos")

SS = 10.0
DURATION = 30.0  # Use None for the remainder of each video.

RECURSIVE = True
OVERWRITE_EXISTING = False
AUTO_DOWNLOAD_EACH_VIDEO = True
DOWNLOAD_OUTPUT_ZIP = False

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"
}
TEMP_DIR = OUTPUT_VIDEO_DIR / "_temporary"

if SS < 0:
    raise ValueError("SS cannot be negative.")
if DURATION is not None and DURATION <= 0:
    raise ValueError("DURATION must be positive or None.")
if AUTO_DOWNLOAD_EACH_VIDEO or DOWNLOAD_OUTPUT_ZIP:
    from google.colab import files as colab_files

if not SOURCE_FACE_PATH.is_file():
    raise FileNotFoundError(f"Source face not found: {SOURCE_FACE_PATH}")
if not INPUT_VIDEO_DIR.is_dir():
    raise NotADirectoryError(f"Input folder not found: {INPUT_VIDEO_DIR}")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

source_image = cv2.imread(str(SOURCE_FACE_PATH))
if source_image is None:
    raise ValueError(f"Could not read source image: {SOURCE_FACE_PATH}")
source_face = get_one_face(source_image)
if source_face is None:
    raise ValueError("No face detected in the source image.")

print(f"✓ Source face: {SOURCE_FACE_PATH}")
print(f"✓ SS: {SS:.3f} seconds")
print(
    "✓ Duration: "
    + (f"{DURATION:.3f} seconds" if DURATION is not None else "remainder of each video")
)

iterator = INPUT_VIDEO_DIR.rglob("*") if RECURSIVE else INPUT_VIDEO_DIR.glob("*")
video_paths = sorted(
    path
    for path in iterator
    if path.is_file()
    and path.suffix.lower() in VIDEO_EXTENSIONS
    and OUTPUT_VIDEO_DIR.resolve() not in path.resolve().parents
)
if not video_paths:
    raise FileNotFoundError(f"No videos found in: {INPUT_VIDEO_DIR}")
print(f"✓ Found {len(video_paths)} video(s)")

segment_parts = []
if SS > 0:
    segment_parts.append(f"ss{SS:g}".replace(".", "p"))
if DURATION is not None:
    segment_parts.append(f"dur{DURATION:g}".replace(".", "p"))
segment_suffix = "_" + "_".join(segment_parts) if segment_parts else ""


def process_video(input_path, output_path, silent_path):
    capture = None
    writer = None
    progress = None
    processed_frames = 0
    fallback_frames = 0

    try:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            fps = 25.0
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid dimensions: {width}x{height}")

        start_frame = int(round(SS * fps))
        if source_frame_count > 0 and start_frame >= source_frame_count:
            raise ValueError(f"SS={SS} is beyond the end of {input_path.name}")

        capture.set(cv2.CAP_PROP_POS_MSEC, SS * 1000.0)

        if DURATION is not None:
            maximum_frames = max(1, int(round(DURATION * fps)))
        elif source_frame_count > 0:
            maximum_frames = max(0, source_frame_count - start_frame)
        else:
            maximum_frames = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        silent_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create: {silent_path}")

        progress = tqdm(
            total=maximum_frames,
            desc=input_path.name,
            unit="frame",
            leave=True,
        )

        while True:
            if maximum_frames is not None and processed_frames >= maximum_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            try:
                output_frame = swap_face(source_face, frame)
                if output_frame is None:
                    output_frame = frame
                    fallback_frames += 1
            except Exception as error:
                output_frame = frame
                fallback_frames += 1
                if fallback_frames <= 3:
                    print(f"\nFrame warning in {input_path.name}: {error}")

            if output_frame.shape[:2] != (height, width):
                output_frame = cv2.resize(
                    output_frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

            writer.write(output_frame)
            processed_frames += 1
            progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        raise RuntimeError(f"No frames processed: {input_path}")

    actual_duration = processed_frames / fps
    ffmpeg_command = ["ffmpeg", "-y", "-i", str(silent_path)]
    if SS > 0:
        ffmpeg_command.extend(["-ss", f"{SS:.6f}"])
    ffmpeg_command.extend(
        [
            "-t", f"{actual_duration:.6f}",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )

    result = subprocess.run(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg failed:\n" + result.stderr[-4000:])
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output not created: {output_path}")

    silent_path.unlink(missing_ok=True)
    return {
        "frames": processed_frames,
        "fallback_frames": fallback_frames,
        "duration": actual_duration,
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }


completed = []
skipped = []
failed = []
downloaded = []

for index, input_path in enumerate(video_paths, start=1):
    relative_path = input_path.relative_to(INPUT_VIDEO_DIR)
    output_path = (
        OUTPUT_VIDEO_DIR
        / relative_path.parent
        / f"{input_path.stem}_face_swapped{segment_suffix}.mp4"
    )
    silent_path = (
        TEMP_DIR
        / relative_path.parent
        / f"{input_path.stem}_silent{segment_suffix}.mp4"
    )

    print(f"\n[{index}/{len(video_paths)}] {relative_path}")

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"↷ Skipped existing: {output_path}")
        skipped.append(output_path)
        continue

    output_path.unlink(missing_ok=True)
    silent_path.unlink(missing_ok=True)

    try:
        information = process_video(input_path, output_path, silent_path)
        completed.append(output_path)

        print(f"✓ Output: {output_path}")
        print(f"  Frames: {information['frames']}")
        print(f"  Duration: {information['duration']:.2f}s")
        print(f"  Fallback frames: {information['fallback_frames']}")
        print(f"  Size: {information['size_mb']:.1f} MB")

        if AUTO_DOWNLOAD_EACH_VIDEO:
            try:
                print(f"↓ Starting download: {output_path.name}")
                colab_files.download(str(output_path))
                downloaded.append(output_path)
            except Exception as download_error:
                print(f"⚠ Automatic download failed: {download_error}")
    except Exception as error:
        silent_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        failed.append((input_path, str(error)))
        print(f"✗ Failed: {input_path}")
        print(f"  Reason: {error}")

try:
    TEMP_DIR.rmdir()
except OSError:
    pass

print("\n" + "=" * 70)
print("BATCH COMPLETE")
print("=" * 70)
print(f"Completed:  {len(completed)}")
print(f"Downloaded: {len(downloaded)}")
print(f"Skipped:    {len(skipped)}")
print(f"Failed:     {len(failed)}")
print(f"Output:     {OUTPUT_VIDEO_DIR}")

if AUTO_DOWNLOAD_EACH_VIDEO:
    print(
        "\nIf only the first download started, allow multiple downloads for "
        "colab.research.google.com in your browser."
    )

if failed:
    print("\nFailed videos:")
    for input_path, error in failed:
        print(f"- {input_path}: {error}")

if DOWNLOAD_OUTPUT_ZIP and completed:
    zip_base = Path("/content/face_swapped_videos")
    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=str(OUTPUT_VIDEO_DIR),
        )
    )
    print(f"\n↓ Downloading ZIP: {zip_path}")
    colab_files.download(str(zip_path))

# %% [code] cell=32 id=batch-video-ffmpeg-pipe
"""CELL: Batch video folder (FFmpeg pipe, up to 30 FPS)"""
# @title Batch video folder (FFmpeg pipe, up to 30 FPS)
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

# Assumes get_one_face() and swap_face() were initialized by earlier notebook cells.

SOURCE_FACE_PATH = Path("/content/vi_0003_portrait.png")
INPUT_VIDEO_DIR = Path("/content/in")
OUTPUT_VIDEO_DIR = Path("/content/outp")

SS = 10.0
DURATION = 10.0  # Use None for the remainder of each video.
MAX_PROCESS_FPS = 30.0  # Preserve lower FPS; cap higher-FPS inputs at this value.
SHORT_VIDEO_SS_POLICY = "start"  # "start" processes short clips from 0; "skip" rejects them.

RECURSIVE = True
OVERWRITE_EXISTING = False
SKIP_ALREADY_PROCESSED = True
AUTO_DOWNLOAD_EACH_VIDEO = False
DOWNLOAD_OUTPUT_ZIP = True

USE_CUDA_DECODE = True
NVENC_ENCODER = "h264_nvenc"
NVENC_PRESET = "p4"
NVENC_CQ = 18

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"
}

if MAX_PROCESS_FPS <= 0:
    raise ValueError("MAX_PROCESS_FPS must be positive.")

if SS < 0:
    raise ValueError("SS cannot be negative.")
if DURATION is not None and DURATION <= 0:
    raise ValueError("DURATION must be positive or None.")
if AUTO_DOWNLOAD_EACH_VIDEO or DOWNLOAD_OUTPUT_ZIP:
    from google.colab import files as colab_files

if not SOURCE_FACE_PATH.is_file():
    raise FileNotFoundError(f"Source face not found: {SOURCE_FACE_PATH}")
if not INPUT_VIDEO_DIR.is_dir():
    raise NotADirectoryError(f"Input folder not found: {INPUT_VIDEO_DIR}")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_MANIFEST_PATH = OUTPUT_VIDEO_DIR / ".face_swap_processed.json"


def processed_input_key(input_path):
    stat = input_path.stat()
    relative = input_path.relative_to(INPUT_VIDEO_DIR).as_posix()
    return f"{relative}|{stat.st_size}|{stat.st_mtime_ns}"


def load_processed_manifest():
    if not PROCESSED_MANIFEST_PATH.is_file():
        return {}
    try:
        payload = json.loads(PROCESSED_MANIFEST_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as error:
        print(f"⚠ Ignoring invalid processed manifest: {error}")
        return {}


def save_processed_manifest():
    temporary_path = Path(str(PROCESSED_MANIFEST_PATH) + ".tmp")
    temporary_path.write_text(
        json.dumps(processed_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(PROCESSED_MANIFEST_PATH)


processed_manifest = load_processed_manifest()

source_image = cv2.imread(str(SOURCE_FACE_PATH))
if source_image is None:
    raise ValueError(f"Could not read source image: {SOURCE_FACE_PATH}")
source_face = get_one_face(source_image)
if source_face is None:
    raise ValueError("No face detected in the source image.")

print(f"✓ Source face: {SOURCE_FACE_PATH}")
print(f"✓ SS: {SS:.3f} seconds")
print(
    "✓ Duration: "
    + (f"{DURATION:.3f} seconds" if DURATION is not None else "remainder of each video")
)
print(f"✓ CUDA decode requested: {USE_CUDA_DECODE}")

iterator = INPUT_VIDEO_DIR.rglob("*") if RECURSIVE else INPUT_VIDEO_DIR.glob("*")
video_paths = sorted(
    path
    for path in iterator
    if path.is_file()
    and path.suffix.lower() in VIDEO_EXTENSIONS
    and OUTPUT_VIDEO_DIR.resolve() not in path.resolve().parents
)
if not video_paths:
    raise FileNotFoundError(f"No videos found in: {INPUT_VIDEO_DIR}")
print(f"✓ Found {len(video_paths)} video(s)")

segment_parts = []
if SS > 0:
    segment_parts.append(f"ss{SS:g}".replace(".", "p"))
if DURATION is not None:
    segment_parts.append(f"dur{DURATION:g}".replace(".", "p"))
segment_suffix = "_" + "_".join(segment_parts) if segment_parts else ""


def parse_fraction(value):
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(input_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed:\n" + result.stderr[-4000:])

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {input_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = parse_fraction(stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = parse_fraction(stream.get("r_frame_rate"))
    if fps <= 0:
        fps = 25.0

    duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = None

    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = None

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid dimensions: {width}x{height}")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
    }


def read_exact(stream, byte_count):
    chunks = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    return data if len(data) == byte_count else b""


def decoder_command(input_path, use_cuda, start_seconds, clip_duration, process_fps):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if use_cuda:
        command.extend(["-hwaccel", "cuda"])
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    command.extend(["-i", str(input_path)])
    if clip_duration is not None:
        command.extend(["-t", f"{clip_duration:.6f}"])
    command.extend(
        [
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            "-vf", f"fps={process_fps:.12g}",
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
    )
    return command


def start_decoder(input_path, use_cuda, start_seconds, clip_duration, process_fps):
    return subprocess.Popen(
        decoder_command(input_path, use_cuda, start_seconds, clip_duration, process_fps),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )


def encoder_command(input_path, output_path, width, height, fps, start_seconds, clip_duration):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.12g}",
        "-i", "pipe:0",
    ]

    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    command.extend(["-t", f"{clip_duration:.6f}", "-i", str(input_path)])
    command.extend(
        [
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-map_metadata", "1",
            "-c:v", NVENC_ENCODER,
            "-preset", NVENC_PRESET,
            "-cq", str(NVENC_CQ),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )
    return command


def process_video(input_path, output_path):
    info = probe_video(input_path)
    width = info["width"]
    height = info["height"]
    source_fps = info["fps"]
    fps = min(source_fps, MAX_PROCESS_FPS)
    print(f"  FPS: {source_fps:.3f} input -> {fps:.3f} processing/output")
    source_duration = info["duration"]

    effective_ss = SS
    if source_duration is not None and effective_ss >= source_duration:
        if SHORT_VIDEO_SS_POLICY == "start":
            print(
                f"⚠ {input_path.name} is shorter than SS={SS}; "
                "processing from 0 seconds instead."
            )
            effective_ss = 0.0
        else:
            raise ValueError(f"SS={SS} is beyond the end of {input_path.name}")

    remaining_duration = (
        None if source_duration is None else max(0.0, source_duration - effective_ss)
    )
    if DURATION is None:
        requested_duration = remaining_duration
    elif remaining_duration is None:
        requested_duration = DURATION
    else:
        requested_duration = min(DURATION, remaining_duration)

    if requested_duration is not None and requested_duration <= 0:
        raise ValueError(f"No duration remains after SS={SS}: {input_path.name}")

    maximum_frames = (
        max(1, int(round(requested_duration * fps)))
        if requested_duration is not None
        else None
    )
    frame_bytes = width * height * 3
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    decoder = None
    encoder = None
    progress = None
    processed_frames = 0
    fallback_frames = 0
    stopped_decoder_early = False

    try:
        decoder = start_decoder(input_path, USE_CUDA_DECODE, effective_ss, requested_duration, fps)
        raw_frame = read_exact(decoder.stdout, frame_bytes)

        if not raw_frame and USE_CUDA_DECODE:
            cuda_error = decoder.stderr.read().decode("utf-8", errors="replace")
            decoder.wait()
            print(f"⚠ CUDA decode unavailable for {input_path.name}; using software decode.")
            if cuda_error.strip():
                print(cuda_error[-1000:])
            decoder = start_decoder(input_path, False, effective_ss, requested_duration, fps)
            raw_frame = read_exact(decoder.stdout, frame_bytes)

        if not raw_frame:
            decode_error = decoder.stderr.read().decode("utf-8", errors="replace")
            decoder.wait()
            raise RuntimeError("FFmpeg produced no frames:\n" + decode_error[-4000:])

        encoder_duration = requested_duration
        if encoder_duration is None:
            if info["frame_count"]:
                encoder_duration = max(1.0 / fps, (info["frame_count"] / source_fps) - effective_ss)
            else:
                encoder_duration = 24 * 60 * 60

        encoder = subprocess.Popen(
            encoder_command(
                input_path,
                output_path,
                width,
                height,
                fps,
                effective_ss,
                encoder_duration,
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

        progress = tqdm(
            total=maximum_frames,
            desc=input_path.name,
            unit="frame",
            leave=True,
        )

        while raw_frame:
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

            try:
                output_frame = swap_face(source_face, frame)
                if output_frame is None:
                    output_frame = frame
                    fallback_frames += 1
            except Exception as error:
                output_frame = frame
                fallback_frames += 1
                if fallback_frames <= 3:
                    print(f"\nFrame warning in {input_path.name}: {error}")

            if output_frame.shape[:2] != (height, width):
                output_frame = cv2.resize(
                    output_frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

            try:
                encoder.stdin.write(np.ascontiguousarray(output_frame).tobytes())
            except BrokenPipeError:
                encode_error = encoder.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError("FFmpeg encoder pipe closed:\n" + encode_error[-4000:])

            processed_frames += 1
            progress.update(1)

            if maximum_frames is not None and processed_frames >= maximum_frames:
                stopped_decoder_early = True
                break

            raw_frame = read_exact(decoder.stdout, frame_bytes)

        if processed_frames == 0:
            raise RuntimeError(f"No frames processed: {input_path}")

        encoder.stdin.close()
        encoder.stdin = None

        if stopped_decoder_early and decoder.poll() is None:
            decoder.terminate()
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder.wait()

        encoder_return_code = encoder.wait()
        encode_error = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder_return_code != 0:
            raise RuntimeError("FFmpeg encode failed:\n" + encode_error[-4000:])

        if not stopped_decoder_early and decoder.returncode != 0:
            decode_error = decoder.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError("FFmpeg decode failed:\n" + decode_error[-4000:])

    finally:
        if progress is not None:
            progress.close()

        if decoder is not None:
            if decoder.stdout is not None and not decoder.stdout.closed:
                decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
                try:
                    decoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    decoder.kill()
                    decoder.wait()

        if encoder is not None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Output not created: {output_path}")

    actual_duration = processed_frames / fps
    return {
        "frames": processed_frames,
        "source_fps": source_fps,
        "output_fps": fps,
        "fallback_frames": fallback_frames,
        "duration": actual_duration,
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }


completed = []
skipped = []
failed = []
downloaded = []

for index, input_path in enumerate(video_paths, start=1):
    relative_path = input_path.relative_to(INPUT_VIDEO_DIR)
    input_key = processed_input_key(input_path)
    output_path = (
        OUTPUT_VIDEO_DIR
        / relative_path.parent
        / f"{input_path.stem}_face_swapped{segment_suffix}.mp4"
    )

    print(f"\n[{index}/{len(video_paths)}] {relative_path}")

    previous_outputs = sorted(
        output_path.parent.glob(f"{input_path.stem}_face_swapped*.mp4")
    )
    manifest_match = input_key in processed_manifest
    should_skip = not OVERWRITE_EXISTING and (
        output_path.exists()
        or (
            SKIP_ALREADY_PROCESSED
            and (manifest_match or bool(previous_outputs))
        )
    )

    if should_skip:
        previous_output = output_path if output_path.exists() else (
            previous_outputs[0] if previous_outputs else None
        )
        print(
            f"↷ Skipped already processed: {relative_path}"
            + (f" -> {previous_output}" if previous_output else "")
        )
        if not manifest_match:
            processed_manifest[input_key] = {
                "input": relative_path.as_posix(),
                "output": str(previous_output) if previous_output else None,
            }
            save_processed_manifest()
        skipped.append(input_path)
        continue

    output_path.unlink(missing_ok=True)

    try:
        information = process_video(input_path, output_path)
        completed.append(output_path)
        processed_manifest[input_key] = {
            "input": relative_path.as_posix(),
            "output": str(output_path),
            "frames": information["frames"],
            "source_fps": information["source_fps"],
            "output_fps": information["output_fps"],
        }
        save_processed_manifest()

        print(f"✓ Output: {output_path}")
        print(f"  Frames: {information['frames']}")
        print(
            f"  FPS: {information['source_fps']:.3f} input -> "
            f"{information['output_fps']:.3f} output"
        )
        print(f"  Duration: {information['duration']:.2f}s")
        print(f"  Fallback frames: {information['fallback_frames']}")
        print(f"  Size: {information['size_mb']:.1f} MB")

        if AUTO_DOWNLOAD_EACH_VIDEO:
            try:
                print(f"↓ Starting download: {output_path.name}")
                colab_files.download(str(output_path))
                downloaded.append(output_path)
            except Exception as download_error:
                print(f"⚠ Automatic download failed: {download_error}")
    except Exception as error:
        output_path.unlink(missing_ok=True)
        failed.append((input_path, str(error)))
        print(f"✗ Failed: {input_path}")
        print(f"  Reason: {error}")

print("\n" + "=" * 70)
print("FFMPEG-PIPE BATCH COMPLETE")
print("=" * 70)
print(f"Completed:  {len(completed)}")
print(f"Downloaded: {len(downloaded)}")
print(f"Skipped:    {len(skipped)}")
print(f"Failed:     {len(failed)}")
print(f"Output:     {OUTPUT_VIDEO_DIR}")

if AUTO_DOWNLOAD_EACH_VIDEO:
    print(
        "\nIf only the first download started, allow multiple downloads for "
        "colab.research.google.com in your browser."
    )

if failed:
    print("\nFailed videos:")
    for input_path, error in failed:
        print(f"- {input_path}: {error}")

if DOWNLOAD_OUTPUT_ZIP and completed:
    zip_base = Path("/content/face_swapped_videos_ffmpeg")
    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=str(OUTPUT_VIDEO_DIR),
        )
    )
    print(f"\n↓ Downloading ZIP: {zip_path}")
    colab_files.download(str(zip_path))
