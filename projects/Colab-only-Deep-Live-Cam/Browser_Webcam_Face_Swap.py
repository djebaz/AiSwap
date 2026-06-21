# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJwcm92ZW5hbmNlIjpbXX0sImtlcm5lbHNwZWMiOnsibmFtZSI6InB5dGhvbjMiLCJkaXNwbGF5X25hbWUiOiJQeXRob24gMyJ9LCJsYW5ndWFnZV9pbmZvIjp7Im5hbWUiOiJweXRob24ifSwibmJmb3JtYXQiOjQsIm5iZm9ybWF0X21pbm9yIjowfQ==

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
