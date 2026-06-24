from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/DeepLiveCamRemote"
APP_STATE = Path.home() / ".deep_live_cam_remote_windows_app.json"


@dataclass
class AppSettings:
    host: str = ""
    port: int = 7860
    drive_root: str = DEFAULT_DRIVE_ROOT
    source_face: str = DEFAULT_DRIVE_ROOT + "/source/source.png"
    photos_input: str = DEFAULT_DRIVE_ROOT + "/photos"
    photos_output: str = DEFAULT_DRIVE_ROOT + "/outputs/photos"
    videos_input: str = DEFAULT_DRIVE_ROOT + "/videos"
    videos_output: str = DEFAULT_DRIVE_ROOT + "/outputs/videos"
    recursive: bool = True
    overwrite: bool = False
    skip_processed: bool = True
    many_faces: bool = False
    enhancer: str = "none"
    max_fps: float = 30.0
    max_width: int = 420
    quality: int = 18
    camera_index: int = 0
    virtual_camera: str = "OBS Virtual Camera"

    @property
    def base_url(self) -> str:
        host = self.host.replace("http://", "").replace("https://", "").strip().strip("/")
        return f"http://{host}:{self.port}"


def load_settings() -> AppSettings:
    if APP_STATE.is_file():
        try:
            data = json.loads(APP_STATE.read_text(encoding="utf-8"))
            return AppSettings(**{**asdict(AppSettings()), **data})
        except Exception:
            pass
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    APP_STATE.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")


class ApiClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.settings.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def job_payload(settings: AppSettings, input_dir: str, output_dir: str) -> dict[str, Any]:
    return {
        "source_face": settings.source_face,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "recursive": settings.recursive,
        "overwrite": settings.overwrite,
        "skip_processed": settings.skip_processed,
        "many_faces": settings.many_faces,
        "enhancer": settings.enhancer,
        "max_fps": settings.max_fps,
        "max_width": settings.max_width,
        "quality": settings.quality,
    }


class LiveWorker(QThread):
    message = Signal(str)
    frame = Signal(bytes)

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            asyncio.run(self._run_live())
        except Exception as exc:
            self.message.emit(f"live stopped: {exc}")

    async def _run_live(self) -> None:
        import cv2
        import websockets
        uri = self.settings.base_url.replace("http://", "ws://") + "/ws/live"
        self.message.emit(f"connecting live websocket: {uri}")
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.settings.camera_index}")
        virtual_cam = None
        try:
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as websocket:
                await websocket.send(json.dumps({"source_face": self.settings.source_face, "enhancer": self.settings.enhancer, "many_faces": self.settings.many_faces, "jpeg_quality": 80}))
                ready = await websocket.recv()
                self.message.emit(f"live backend: {ready}")
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.03)
                        continue
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    await websocket.send(encoded.tobytes())
                    reply = await websocket.recv()
                    if isinstance(reply, str):
                        self.message.emit(reply)
                        continue
                    self.frame.emit(reply)
                    if virtual_cam is None:
                        try:
                            import pyvirtualcam
                            decoded = cv2.imdecode(__import__('numpy').frombuffer(reply, dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
                            h, w = decoded.shape[:2]
                            virtual_cam = pyvirtualcam.Camera(width=w, height=h, fps=20, device=self.settings.virtual_camera or None)
                            self.message.emit(f"virtual camera opened: {virtual_cam.device}")
                        except Exception as exc:
                            self.message.emit(f"virtual camera unavailable: {exc}")
                            virtual_cam = False
                    if virtual_cam:
                        import numpy as np
                        decoded = cv2.imdecode(np.frombuffer(reply, dtype=np.uint8), cv2.IMREAD_COLOR)
                        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        virtual_cam.send(rgb)
                        virtual_cam.sleep_until_next_frame()
        finally:
            cap.release()
            if virtual_cam and hasattr(virtual_cam, "close"):
                virtual_cam.close()
            self.message.emit("live worker stopped")


class PollWorker(QThread):
    message = Signal(str)
    finished_status = Signal(str)

    def __init__(self, client: ApiClient, job_id: str):
        super().__init__()
        self.client = client
        self.job_id = job_id
        self._seen = 0
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            try:
                payload = self.client.request_json("GET", f"/jobs/{self.job_id}", timeout=5)
                logs = payload.get("logs") or []
                for line in logs[self._seen:]:
                    self.message.emit(str(line))
                self._seen = len(logs)
                status = payload.get("status", "unknown")
                if status not in {"queued", "running"}:
                    self.finished_status.emit(status)
                    return
            except Exception as exc:
                self.message.emit(f"poll error: {exc}")
            time.sleep(1.0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.client = ApiClient(self.settings)
        self.poller: PollWorker | None = None
        self.live_worker: LiveWorker | None = None
        self.active_job_id: str | None = None
        self.setWindowTitle("Deep-Live-Cam Remote Controller")
        self.resize(980, 720)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.log_box = QTextEdit(readOnly=True)
        self._build_setup_tab()
        self._build_photos_tab()
        self._build_videos_tab()
        self._build_live_tab()
        self.tabs.addTab(self.log_box, "Logs")

    def log(self, text: str) -> None:
        self.log_box.append(text)

    def sync_settings(self) -> None:
        self.settings.host = self.host.text().strip()
        self.settings.port = int(self.port.value())
        self.settings.drive_root = self.drive_root.text().strip()
        self.settings.source_face = self.source_face.text().strip()
        self.settings.photos_input = self.photos_input.text().strip()
        self.settings.photos_output = self.photos_output.text().strip()
        self.settings.videos_input = self.videos_input.text().strip()
        self.settings.videos_output = self.videos_output.text().strip()
        self.settings.recursive = self.recursive.isChecked()
        self.settings.overwrite = self.overwrite.isChecked()
        self.settings.skip_processed = self.skip_processed.isChecked()
        self.settings.many_faces = self.many_faces.isChecked()
        self.settings.enhancer = self.enhancer.currentText()
        self.settings.max_fps = float(self.max_fps.value())
        self.settings.max_width = int(self.max_width.value())
        self.settings.quality = int(self.quality.value())
        self.settings.camera_index = int(self.camera_index.value())
        self.settings.virtual_camera = self.virtual_camera.text().strip()
        save_settings(self.settings)

    def _build_setup_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.host = QLineEdit(self.settings.host)
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(self.settings.port)
        self.drive_root = QLineEdit(self.settings.drive_root)
        form.addRow("Tailscale host/IP", self.host)
        form.addRow("API port", self.port)
        form.addRow("Drive root", self.drive_root)
        layout.addLayout(form)
        self.setup_help = QTextEdit(readOnly=True)
        self.setup_help.setPlainText(
            "Colab setup checklist:\n"
            "1. Open google-colab/Deep_Live_Cam_Remote_Batch.ipynb.\n"
            "2. Run Install and initialize.\n"
            "3. Mount Drive and use /content/drive/MyDrive/DeepLiveCamRemote.\n"
            "4. Run the Remote API server cell.\n"
            "5. Start Tailscale in Colab and copy the Tailscale IP here.\n"
        )
        layout.addWidget(self.setup_help)
        row = QHBoxLayout()
        btn = QPushButton("Check connection"); btn.clicked.connect(self.check_connection)
        save = QPushButton("Save settings"); save.clicked.connect(lambda: (self.sync_settings(), self.log("settings saved")))
        row.addWidget(btn); row.addWidget(save); row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(tab, "Setup")

    def _common_group(self) -> QGroupBox:
        box = QGroupBox("Common options")
        form = QFormLayout(box)
        self.source_face = QLineEdit(self.settings.source_face)
        self.recursive = QCheckBox(); self.recursive.setChecked(self.settings.recursive)
        self.overwrite = QCheckBox(); self.overwrite.setChecked(self.settings.overwrite)
        self.skip_processed = QCheckBox(); self.skip_processed.setChecked(self.settings.skip_processed)
        self.many_faces = QCheckBox(); self.many_faces.setChecked(self.settings.many_faces)
        self.enhancer = QComboBox(); self.enhancer.addItems(["none", "gfpgan", "gpen256", "gpen512"]); self.enhancer.setCurrentText(self.settings.enhancer)
        form.addRow("Source face path", self.source_face)
        form.addRow("Recursive", self.recursive)
        form.addRow("Overwrite", self.overwrite)
        form.addRow("Skip processed", self.skip_processed)
        form.addRow("Many faces", self.many_faces)
        form.addRow("Enhancer", self.enhancer)
        return box

    def _build_photos_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        layout.addWidget(self._common_group())
        form = QFormLayout()
        self.photos_input = QLineEdit(self.settings.photos_input)
        self.photos_output = QLineEdit(self.settings.photos_output)
        form.addRow("Photos input path", self.photos_input)
        form.addRow("Photos output path", self.photos_output)
        layout.addLayout(form)
        btn = QPushButton("Start photo batch"); btn.clicked.connect(self.start_photos)
        layout.addWidget(btn); layout.addStretch(1)
        self.tabs.addTab(tab, "Photos")

    def _build_videos_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.videos_input = QLineEdit(self.settings.videos_input)
        self.videos_output = QLineEdit(self.settings.videos_output)
        self.max_fps = QDoubleSpinBox(); self.max_fps.setRange(1, 120); self.max_fps.setValue(self.settings.max_fps)
        self.max_width = QSpinBox(); self.max_width.setRange(64, 4096); self.max_width.setValue(self.settings.max_width)
        self.quality = QSpinBox(); self.quality.setRange(0, 51); self.quality.setValue(self.settings.quality)
        form.addRow("Videos input path", self.videos_input)
        form.addRow("Videos output path", self.videos_output)
        form.addRow("Max FPS", self.max_fps)
        form.addRow("Max width", self.max_width)
        form.addRow("Quality", self.quality)
        layout.addLayout(form)
        btn = QPushButton("Start video batch"); btn.clicked.connect(self.start_videos)
        cancel = QPushButton("Graceful cancel active job"); cancel.clicked.connect(self.cancel_job)
        row = QHBoxLayout(); row.addWidget(btn); row.addWidget(cancel); row.addStretch(1)
        layout.addLayout(row); layout.addStretch(1)
        self.tabs.addTab(tab, "Videos")

    def _build_live_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.camera_index = QSpinBox(); self.camera_index.setRange(0, 20); self.camera_index.setValue(self.settings.camera_index)
        self.virtual_camera = QLineEdit(self.settings.virtual_camera)
        form.addRow("Camera index", self.camera_index)
        form.addRow("Virtual camera", self.virtual_camera)
        layout.addLayout(form)
        self.live_note = QLabel("Live sends webcam JPEG frames to ws://HOST:PORT/ws/live, previews returned frames, and opens the configured virtual camera when pyvirtualcam can find it.")
        self.live_note.setWordWrap(True)
        layout.addWidget(self.live_note)
        self.live_preview = QLabel("Live preview")
        self.live_preview.setAlignment(Qt.AlignCenter)
        self.live_preview.setMinimumHeight(360)
        layout.addWidget(self.live_preview)
        row = QHBoxLayout()
        start = QPushButton("Start live")
        stop = QPushButton("Stop live")
        start.clicked.connect(self.start_live)
        stop.clicked.connect(self.stop_live)
        row.addWidget(start); row.addWidget(stop); row.addStretch(1)
        layout.addLayout(row); layout.addStretch(1)
        self.tabs.addTab(tab, "Live")

    def check_connection(self) -> None:
        self.sync_settings()
        try:
            payload = self.client.request_json("GET", "/health")
            self.log("health: " + json.dumps(payload, indent=2))
        except Exception as exc:
            self.log(f"health failed: {exc}")

    def start_job(self, endpoint: str, payload: dict[str, Any]) -> None:
        self.sync_settings()
        try:
            response = self.client.request_json("POST", endpoint, payload)
            self.active_job_id = response.get("job_id")
            self.log(f"started {endpoint}: {response}")
            if self.active_job_id:
                if self.poller:
                    self.poller.stop()
                self.poller = PollWorker(self.client, self.active_job_id)
                self.poller.message.connect(self.log)
                self.poller.finished_status.connect(lambda status: self.log(f"job finished: {status}"))
                self.poller.start()
                self.tabs.setCurrentWidget(self.log_box)
        except Exception as exc:
            self.log(f"start failed: {exc}")

    def start_photos(self) -> None:
        self.sync_settings()
        self.start_job("/jobs/photos", job_payload(self.settings, self.settings.photos_input, self.settings.photos_output))

    def start_videos(self) -> None:
        self.sync_settings()
        self.start_job("/jobs/videos", job_payload(self.settings, self.settings.videos_input, self.settings.videos_output))

    def cancel_job(self) -> None:
        self.sync_settings()
        if not self.active_job_id:
            self.log("no active job")
            return
        try:
            payload = self.client.request_json("POST", "/jobs/cancel", {"job_id": self.active_job_id})
            self.log("cancel: " + json.dumps(payload))
        except Exception as exc:
            self.log(f"cancel failed: {exc}")

    def start_live(self) -> None:
        self.sync_settings()
        if self.live_worker and self.live_worker.isRunning():
            self.log("live already running")
            return
        self.live_worker = LiveWorker(self.settings)
        self.live_worker.message.connect(self.log)
        self.live_worker.frame.connect(self.update_live_preview)
        self.live_worker.start()

    def stop_live(self) -> None:
        if self.live_worker:
            self.live_worker.stop()
            self.log("live stop requested")

    def update_live_preview(self, jpeg_bytes: bytes) -> None:
        image = QImage.fromData(jpeg_bytes, "JPG")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(self.live_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.live_preview.setPixmap(pixmap)


def main() -> int:
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
