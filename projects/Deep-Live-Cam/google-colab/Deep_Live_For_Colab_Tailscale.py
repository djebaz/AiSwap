# Auto-generated from notebook; keep markers for round-trip
# Markers + docstring headers are required for ipynb reconstruction
# NOTEBOOK_META_B64=eyJjb2xhYiI6eyJjb2xsYXBzZWRfc2VjdGlvbnMiOlsib3ZlcnZpZXciLCJydW50aW1lIiwicmVzdGFydC1ub3RlIiwic2VydmVyLW5vdGVzIiwidGFpbHNjYWxlLW5vdGVzIiwid2luZG93cy11c2FnZSJdfSwia2VybmVsc3BlYyI6eyJuYW1lIjoicHl0aG9uMyIsImxhbmd1YWdlIjoicHl0aG9uIiwiZGlzcGxheV9uYW1lIjoiUHl0aG9uIDMifSwibGFuZ3VhZ2VfaW5mbyI6eyJuYW1lIjoicHl0aG9uIn0sIm5iZm9ybWF0Ijo0LCJuYmZvcm1hdF9taW5vciI6NX0=

# %% [markdown] cell=0 id=overview
"""MARKDOWN
# Deep Live Cam Remote GPU via Tailscale

This notebook runs the face-swap model on a Colab GPU and connects privately to the modified Windows client over Tailscale.

Highlights:

- No Ngrok, FRP, public TCP endpoint, or payment-card verification.
- Ports `5555`, `5556`, and `5557` remain private to your tailnet.
- Restartable ZMQ workers recover from interrupted Windows clients.
- Five-second socket timeouts prevent permanently wedged result threads.
- Four-megabyte chunks reduce round-trip overhead.

Use this only with media you are authorized to process. The local NSFW detector was disabled in the companion Windows fork.
"""ENDMARKDOWN

# %% [markdown] cell=1 id=runtime
"""MARKDOWN
## 1. Select a GPU runtime

Choose **Runtime → Change runtime type → T4 GPU** before continuing.
"""ENDMARKDOWN

# %% [code] cell=2 id=install
"""CELL: Install dependencies"""
# @title Install dependencies
%pip install -q --upgrade pip setuptools wheel
%pip install -q --no-cache-dir "numpy<2" insightface==0.7.3 pyzmq tqdm
%pip uninstall -y onnxruntime onnxruntime-gpu
%pip install -q --no-cache-dir --upgrade onnxruntime-gpu
!apt-get update -qq
!apt-get install -y -qq ffmpeg

# %% [markdown] cell=3 id=restart-note
"""MARKDOWN
### Restart once after installation

After the installation cell finishes, choose **Runtime → Restart session**, then continue below. Do not rerun the installation cell unless the Colab VM is replaced.
"""ENDMARKDOWN

# %% [code] cell=4 id=gpu-check
"""CELL: Verify CUDA and ONNX Runtime"""
# @title Verify CUDA and ONNX Runtime
import torch
import onnxruntime as ort

print("PyTorch CUDA:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("ONNX Runtime:", ort.__version__)
print("ONNX Runtime device:", ort.get_device())
print("ONNX Runtime providers:", ort.get_available_providers())

assert torch.cuda.is_available(), "Colab GPU is unavailable. Select a T4 GPU runtime."
assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "CUDAExecutionProvider is unavailable. Restart the session after installing dependencies."
)

# %% [code] cell=5 id=models
"""CELL: Download the GPU models"""
# @title Download the GPU models
from pathlib import Path
import urllib.request

MODEL_DIR = Path("/content/deepfakecollab/Model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SWAPPER_PATH = MODEL_DIR / "inswapper_128.onnx"
ENHANCER_PATH = MODEL_DIR / "GFPGANv1.4.onnx"

if not SWAPPER_PATH.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx",
        SWAPPER_PATH,
    )
if not ENHANCER_PATH.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",
        ENHANCER_PATH,
    )
print("Face-swap model:", SWAPPER_PATH, SWAPPER_PATH.stat().st_size, "bytes")
print("Face-enhancer model:", ENHANCER_PATH, ENHANCER_PATH.stat().st_size, "bytes")

# %% [code] cell=6 id=face-runtime
"""CELL: Initialize GPU face processing"""
# @title Initialize GPU face processing
import cv2
import insightface
import numpy as np
import onnxruntime as ort
import threading

ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
print("Using providers:", ORT_PROVIDERS)

FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=ORT_PROVIDERS)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
_FACE_SWAPPER = None
_FACE_ENHANCER = None
_MODEL_LOCK = threading.Lock()


def get_face_swapper():
    global _FACE_SWAPPER
    with _MODEL_LOCK:
        if _FACE_SWAPPER is None:
            _FACE_SWAPPER = insightface.model_zoo.get_model(
                str(SWAPPER_PATH), providers=ORT_PROVIDERS
            )
    return _FACE_SWAPPER


def get_face_enhancer():
    global _FACE_ENHANCER
    with _MODEL_LOCK:
        if _FACE_ENHANCER is None:
            _FACE_ENHANCER = ort.InferenceSession(str(ENHANCER_PATH), providers=ORT_PROVIDERS)
            assert _FACE_ENHANCER.get_providers()[0] == "CUDAExecutionProvider", (
                f"GFPGAN fell back to {_FACE_ENHANCER.get_providers()}"
            )
    return _FACE_ENHANCER


def get_faces(frame):
    if frame is None:
        return []
    return FACE_ANALYSER.get(frame) or []


def get_one_face(frame):
    faces = get_faces(frame)
    return min(faces, key=lambda face: face.bbox[0]) if faces else None


def enhance_face(frame, face):
    session = get_face_enhancer()
    size = int(session.get_inputs()[0].shape[-1])
    template = np.array([
        [192.98138, 239.94708], [318.90277, 240.19360],
        [256.63416, 314.01935], [201.26117, 371.41043],
        [313.08905, 371.15118],
    ], dtype=np.float32) * (size / 512.0)
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(face.kps, np.float32), template)
    if matrix is None:
        return frame
    crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    tensor = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    restored = np.transpose(output, (1, 2, 0))
    restored = np.clip((restored + 1.0) * 127.5, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
    inverse = cv2.invertAffineTransform(matrix)
    pasted = cv2.warpAffine(restored, inverse, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpAffine(np.full((size, size), 255, np.uint8), inverse, (frame.shape[1], frame.shape[0]))
    mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, size / 32.0)).astype(np.float32) / 255.0
    return (pasted * mask[..., None] + frame * (1.0 - mask[..., None])).astype(np.uint8)


def process_frame(source_frame, target_frame, many_faces=False, enhance=False):
    source_face = get_one_face(source_frame)
    if source_face is None:
        raise ValueError("No face detected in the source image")

    target_faces = get_faces(target_frame)
    if not target_faces:
        raise ValueError("No face detected in the target image")
    if not many_faces:
        target_faces = [min(target_faces, key=lambda face: face.bbox[0])]

    result = target_frame.copy()
    swapper = get_face_swapper()
    for target_face in target_faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)
    if enhance:
        for result_face in get_faces(result):
            result = enhance_face(result, result_face)
    return result

swapper = get_face_swapper()
assert swapper.session.get_providers()[0] == "CUDAExecutionProvider", (
    f"Face swapper fell back to {swapper.session.get_providers()}"
)
for model in FACE_ANALYSER.models.values():
    if hasattr(model, "session"):
        assert model.session.get_providers()[0] == "CUDAExecutionProvider", (
            f"Face analyser fell back to {model.session.get_providers()}"
        )
print("GPU face runtime initialized; all ONNX sessions use CUDAExecutionProvider")

# %% [markdown] cell=7 id=server-notes
"""MARKDOWN
## 2. Start the restartable remote server

The Windows client protocol uses:

- `5555`: source image upload
- `5556`: target image upload
- `5557`: processed result download

Running the **Start or reset remote server** cell safely replaces any prior workers in this session.
"""ENDMARKDOWN

# %% [code] cell=8 id=server-implementation
"""CELL: Define restartable ZMQ server"""
# @title Define restartable ZMQ server
import math
import queue
import threading
import time
import traceback
import zmq
import numpy as np


class RemoteSwapServer:
    def __init__(self, host="127.0.0.1", timeout_ms=5000, chunk_bytes=4 * 1024 * 1024):
        self.host = host
        self.timeout_ms = timeout_ms
        self.chunk_bytes = chunk_bytes
        self.context = zmq.Context.instance()
        self.stop_event = threading.Event()
        self.source_queue = queue.Queue(maxsize=1)
        self.target_queue = queue.Queue(maxsize=1)
        self.threads = []

    @staticmethod
    def _put_latest(destination, value):
        while True:
            try:
                destination.put_nowait(value)
                return
            except queue.Full:
                try:
                    destination.get_nowait()
                except queue.Empty:
                    pass

    def _socket(self, socket_type, port):
        socket = self.context.socket(socket_type)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.bind(f"tcp://{self.host}:{port}")
        return socket

    @staticmethod
    def _receive_array(socket, dtype_key, shape_key):
        metadata = socket.recv_json()
        total_chunks = int(metadata["total_chunk"])
        socket.send_string("ACK")
        payload = bytearray()
        for index in range(total_chunks):
            payload.extend(socket.recv())
            socket.send_string(f"ACK {index + 1}/{total_chunks}")
        if socket.recv() != b"END":
            raise RuntimeError("Missing END marker")
        socket.send_string("Final ACK")
        array = np.frombuffer(payload, dtype=np.dtype(metadata[dtype_key]))
        array = array.reshape(metadata[shape_key]).copy()
        return array, metadata

    def _receive_loop(self, port, dtype_key, shape_key, destination, label):
        while not self.stop_event.is_set():
            socket = None
            try:
                socket = self._socket(zmq.REP, port)
                while not self.stop_event.is_set():
                    try:
                        array, metadata = self._receive_array(socket, dtype_key, shape_key)
                        self._put_latest(destination, (array, metadata))
                        print(f"Received {label}: {array.shape}")
                    except zmq.Again:
                        continue
                    except Exception as exception:
                        print(f"Resetting {label} socket: {exception}")
                        break
            except zmq.ZMQError as exception:
                if not self.stop_event.is_set():
                    print(f"Unable to bind {label} port {port}: {exception}")
                    time.sleep(1)
            finally:
                if socket is not None:
                    socket.close(0)

    def _send_result(self, result):
        socket = self._socket(zmq.REQ, 5557)
        try:
            result = np.ascontiguousarray(result)
            payload = memoryview(result).cast("B")
            total_chunks = max(1, math.ceil(len(payload) / self.chunk_bytes))
            socket.send_json({
                "dtype_source": str(result.dtype),
                "shape_source": result.shape,
                "size": "640x480",
                "fps": "60",
                "total_chunk": total_chunks,
            })
            socket.recv_string()
            for index in range(total_chunks):
                start = index * self.chunk_bytes
                socket.send(payload[start:start + self.chunk_bytes])
                socket.recv_string()
            socket.send(b"END")
            socket.recv_string()
            print(f"Returned result: {result.shape} in {total_chunks} chunk(s)")
        finally:
            socket.close(0)

    def _result_loop(self):
        while not self.stop_event.is_set():
            try:
                source, source_metadata = self.source_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                target, _ = self.target_queue.get(timeout=30)
                result = process_frame(
                    source,
                    target,
                    bool(source_metadata.get("manyface", False)),
                    bool(source_metadata.get("enhance", False)),
                )
                self._send_result(result)
            except queue.Empty:
                print("Target upload timed out; dropping unmatched source")
            except zmq.Again:
                print("Result client timed out; result socket reset and server remains usable")
            except Exception as exception:
                print(f"Remote processing failed: {exception}")
                traceback.print_exc()

    def start(self):
        thread_specs = [
            (self._receive_loop, (5555, "dtype_source", "shape_source", self.source_queue, "source")),
            (self._receive_loop, (5556, "dtype_temp", "shape_temp", self.target_queue, "target")),
            (self._result_loop, ()),
        ]
        self.threads = [
            threading.Thread(target=target, args=args, daemon=True)
            for target, args in thread_specs
        ]
        for thread in self.threads:
            thread.start()
        time.sleep(1)
        return self.health()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=7)

    def health(self):
        status = {
            "source_thread": bool(self.threads and self.threads[0].is_alive()),
            "target_thread": bool(len(self.threads) > 1 and self.threads[1].is_alive()),
            "result_thread": bool(len(self.threads) > 2 and self.threads[2].is_alive()),
            "source_queue": self.source_queue.qsize(),
            "target_queue": self.target_queue.qsize(),
        }
        print(status)
        return status

print("Restartable ZMQ server defined")

# %% [code] cell=9 id=start-server
"""CELL: Start or reset remote server"""
# @title Start or reset remote server
try:
    REMOTE_SERVER.stop()
except (NameError, AttributeError):
    pass
try:
    LIVE_SERVER.stop()
except (NameError, AttributeError):
    pass

REMOTE_SERVER = RemoteSwapServer()
REMOTE_SERVER.start()

# %% [markdown] cell=10 id=live-notes
"""MARKDOWN
## 3. Live webcam to Windows virtual camera

Live mode uses the same private ports but switches `5556` and `5557` to low-latency MPEG-TS streams. Stop image mode, then start live mode. The Windows client uploads the selected source face on `5555`, streams the physical webcam to `5556`, and publishes returned frames through OBS Virtual Camera (or another pyvirtualcam backend).
"""ENDMARKDOWN

# %% [code] cell=11 id=live-server-implementation
"""CELL: Define live GPU stream server"""
# @title Define live GPU stream server
import subprocess


class LiveSwapServer:
    def __init__(self, width=960, height=540, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.stop_event = threading.Event()
        self.thread = None
        self.input_process = None
        self.output_process = None
        self.source_socket = None
        self.last_error = None

    def _run(self):
        try:
            context = zmq.Context.instance()
            self.source_socket = context.socket(zmq.REP)
            self.source_socket.setsockopt(zmq.LINGER, 0)
            self.source_socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.source_socket.bind("tcp://127.0.0.1:5555")
            print("Live server waiting for the source face on port 5555...")
            source = metadata = None
            while not self.stop_event.is_set() and source is None:
                try:
                    source, metadata = RemoteSwapServer._receive_array(
                        self.source_socket, "dtype_source", "shape_source"
                    )
                except zmq.Again:
                    continue
            if source is None:
                return

            enhance = bool(metadata.get("enhance", False))
            many_faces = bool(metadata.get("manyface", False))
            input_command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-probesize", "32", "-analyzeduration", "0",
                "-i", "tcp://127.0.0.1:5556?listen=1",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "pipe:1",
            ]
            output_command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-tune", "zerolatency",
                "-g", "1", "-bf", "0", "-f", "mpegts",
                "tcp://127.0.0.1:5557?listen=1",
            ]
            self.input_process = subprocess.Popen(input_command, stdout=subprocess.PIPE)
            self.output_process = subprocess.Popen(output_command, stdin=subprocess.PIPE)
            frame_bytes = self.width * self.height * 3
            print("Live GPU stream ready; waiting for the Windows webcam streams...")
            while not self.stop_event.is_set():
                raw = self.input_process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise EOFError("Windows webcam input stream closed")
                frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
                result = process_frame(source, frame, many_faces, enhance)
                self.output_process.stdin.write(np.ascontiguousarray(result).tobytes())
                self.output_process.stdin.flush()
        except Exception as exception:
            if not self.stop_event.is_set():
                self.last_error = repr(exception)
                traceback.print_exc()
        finally:
            for process in (self.input_process, self.output_process):
                if process is not None and process.poll() is None:
                    process.terminate()
            if self.source_socket is not None:
                self.source_socket.close(0)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(1)
        return self.health()

    def stop(self):
        self.stop_event.set()
        for process in (self.input_process, self.output_process):
            if process is not None and process.poll() is None:
                process.terminate()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def health(self):
        status = {"live_thread": bool(self.thread and self.thread.is_alive()), "error": self.last_error}
        print(status)
        return status

print("Live stream server defined")

# %% [code] cell=12 id=start-live-server
"""CELL: Switch to live webcam server"""
# @title Switch to live webcam server
try:
    REMOTE_SERVER.stop()
except (NameError, AttributeError):
    pass
try:
    LIVE_SERVER.stop()
except (NameError, AttributeError):
    pass

LIVE_SERVER = LiveSwapServer(width=960, height=540, fps=30)
LIVE_SERVER.start()

# %% [markdown] cell=13 id=tailscale-notes
"""MARKDOWN
## 3. Connect Colab to Tailscale

Create an **ephemeral, one-off auth key** in the Tailscale admin console. Add it to Colab Secrets as `TAILSCALE_AUTHKEY` and enable notebook access.

The authentication key is only read if the existing Tailscale state is not already connected.
"""ENDMARKDOWN

# %% [code] cell=14 id=install-tailscale
"""CELL: Install Tailscale"""
# @title Install Tailscale
!command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
!tailscale version

# %% [code] cell=15 id=connect-tailscale
"""CELL: Start Tailscale and expose private TCP ports"""
# @title Start Tailscale and expose private TCP ports
import json
import os
from pathlib import Path
import subprocess
import time

TAILSCALE_SOCKET = "/var/run/tailscale/tailscaled.sock"
Path("/var/run/tailscale").mkdir(parents=True, exist_ok=True)

status = subprocess.run(
    ["tailscale", "--socket", TAILSCALE_SOCKET, "status", "--json"],
    capture_output=True,
    text=True,
)
if status.returncode != 0:
    TAILSCALED_PROCESS = subprocess.Popen(
        [
            "tailscaled",
            "--tun=userspace-networking",
            "--state=/tmp/tailscaled.state",
            f"--socket={TAILSCALE_SOCKET}",
        ],
        stdout=open("/tmp/tailscaled.log", "a"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)

status = subprocess.run(
    ["tailscale", "--socket", TAILSCALE_SOCKET, "status", "--json"],
    capture_output=True,
    text=True,
)
backend_state = None
if status.returncode == 0:
    backend_state = json.loads(status.stdout).get("BackendState")

if backend_state != "Running":
    from google.colab import userdata

    auth_key = userdata.get("TAILSCALE_AUTHKEY")
    if not auth_key:
        raise RuntimeError("Add TAILSCALE_AUTHKEY to Colab Secrets")
    key_path = Path("/tmp/tailscale-authkey")
    key_path.write_text(auth_key)
    key_path.chmod(0o600)
    try:
        subprocess.run(
            [
                "tailscale", "--socket", TAILSCALE_SOCKET, "up",
                f"--auth-key=file:{key_path}",
                "--hostname=deep-live-colab",
            ],
            check=True,
        )
    finally:
        key_path.unlink(missing_ok=True)

for port in (5555, 5556, 5557):
    subprocess.run(
        [
            "tailscale", "--socket", TAILSCALE_SOCKET,
            "serve", "--bg", f"--tcp={port}", f"tcp://127.0.0.1:{port}",
        ],
        check=True,
    )

tailscale_ip = subprocess.check_output(
    ["tailscale", "--socket", TAILSCALE_SOCKET, "ip", "-4"], text=True
).strip()
print("Tailscale IP:", tailscale_ip)
print(
    "Windows command:\n"
    f"python .\\run.py --frame-processor remote_processor --remote-host {tailscale_ip}"
)

# %% [code] cell=16 id=health
"""CELL: Check server and Tailscale health"""
# @title Check server and Tailscale health
if "LIVE_SERVER" in globals() and LIVE_SERVER.thread and LIVE_SERVER.thread.is_alive():
    LIVE_SERVER.health()
else:
    REMOTE_SERVER.health()
subprocess.run(["tailscale", "--socket", TAILSCALE_SOCKET, "status"])
subprocess.run(["tailscale", "--socket", TAILSCALE_SOCKET, "serve", "status"])

# %% [markdown] cell=17 id=windows-usage
"""MARKDOWN
## 4. Run the Windows client

Interactive GUI:

```powershell
python .\run.py --frame-processor remote_processor --remote-host TAILSCALE_IP
```

Headless image URLs:

```powershell
python .\run.py `
  --frame-processor remote_processor `
  --remote-host TAILSCALE_IP `
  --source 'https://example.com/source.jpg' `
  --target 'https://example.com/target.jpg' `
  --output '.\output.png'
```

If Windows disconnects during a response, this notebook times out and automatically resets the result socket. For a manual reset, rerun **Start or reset remote server**; there is no need to restart the entire Colab session.

Live physical webcam to virtual webcam (run after **Switch to live webcam server**):

```powershell
python .\run.py `
  --frame-processor remote_processor `
  --remote-host TAILSCALE_IP `
  --source '.\source-face.jpg' `
  --live `
  --virtual-camera 'OBS Virtual Camera'
```

Install OBS Studio and start **OBS Virtual Camera** once so the Windows device exists. Then select `OBS Virtual Camera` as the camera in Zoom, Discord, Teams, or a browser. Add `--remote-face-enhancer` to run GFPGAN on the Colab GPU; it improves quality but lowers live FPS.
"""ENDMARKDOWN
