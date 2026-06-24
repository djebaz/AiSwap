from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QListWidgetItem

from windows_app import app as base


class OutputTaskWorker(QThread):
    succeeded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, task_id: str, task: Callable[[], object]):
        super().__init__()
        self.task_id = task_id
        self.task = task

    def run(self) -> None:
        try:
            self.succeeded.emit(self.task_id, self.task())
        except Exception as exc:
            self.failed.emit(self.task_id, str(exc))


def _ensure_output_worker_state(window: base.MainWindow) -> None:
    if not hasattr(window, "output_workers"):
        window.output_workers = {}
    if not hasattr(window, "output_refresh_task_id"):
        window.output_refresh_task_id = ""
    if not hasattr(window, "output_preview_task_id"):
        window.output_preview_task_id = ""
    if not hasattr(window, "output_download_task_id"):
        window.output_download_task_id = ""
    if not hasattr(window, "output_health_task_id"):
        window.output_health_task_id = ""


def _start_output_task(
    window: base.MainWindow,
    status: str,
    task: Callable[[], object],
    on_success: Callable[[str, object], None],
    on_failure: Callable[[str, str], None],
) -> str:
    _ensure_output_worker_state(window)
    task_id = uuid.uuid4().hex
    worker = OutputTaskWorker(task_id, task)
    window.output_workers[task_id] = worker
    window.output_status.setText(status)
    worker.succeeded.connect(on_success)
    worker.failed.connect(on_failure)
    worker.finished.connect(lambda task_id=task_id: window.output_workers.pop(task_id, None))
    worker.start()
    return task_id


def refresh_outputs(self: base.MainWindow) -> None:
    self.sync_settings()
    _ensure_output_worker_state(self)
    kind = self.outputs_kind.currentText()
    self.outputs_list.clear()
    self.outputs_list.setEnabled(False)
    self.output_files = []
    self.stop_output_video()

    def fetch() -> dict[str, Any]:
        return self.client.request_json("GET", f"/outputs/{kind}", timeout=5.0)

    def succeeded(task_id: str, payload: object) -> None:
        if task_id != self.output_refresh_task_id:
            return
        self.outputs_list.setEnabled(True)
        self.output_files = list((payload if isinstance(payload, dict) else {}).get("files") or [])
        for item in self.output_files:
            label = f"[{item.get('source')}] {item.get('relative_path')} ({base.format_size(item.get('size'))})"
            self.outputs_list.addItem(QListWidgetItem(label))
        self.output_status.setText(f"{len(self.output_files)} {kind} output file(s)")
        if self.output_files:
            self.outputs_list.setCurrentRow(0)
        else:
            self.output_preview.setPixmap(QPixmap())
            self.output_preview.setText("No remote outputs found")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_refresh_task_id:
            return
        self.outputs_list.setEnabled(True)
        self.output_status.setText(f"refresh failed: {error}")
        self.log(f"outputs refresh failed: {error}")

    self.output_refresh_task_id = _start_output_task(self, "Refreshing outputs...", fetch, succeeded, failed)


def show_output_at(self: base.MainWindow, index: int) -> None:
    if index < 0 or index >= len(self.output_files):
        return
    _ensure_output_worker_state(self)
    item = dict(self.output_files[index])
    kind = self.outputs_kind.currentText()
    path = str(item.get("download_path") or "")
    if not path:
        self.output_status.setText("selected output has no download path")
        return
    self.stop_output_video()
    if kind == "photos":
        self.output_preview.setPixmap(QPixmap())
        self.output_preview.setText("Loading photo preview...")

        def fetch_photo() -> bytes:
            return self.client.download_bytes(path, timeout=20.0)

        def photo_ready(task_id: str, data: object) -> None:
            if task_id != self.output_preview_task_id:
                return
            if self.output_video is not None:
                self.output_video.hide()
            self.output_preview.show()
            image = QImage.fromData(data if isinstance(data, bytes) else bytes(data))
            if image.isNull():
                self.output_status.setText("preview failed: downloaded image could not be decoded")
                return
            pixmap = QPixmap.fromImage(image).scaled(self.output_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.output_preview.setPixmap(pixmap)
            self.output_status.setText(f"Showing {item.get('relative_path')} from {item.get('source')}")

        def photo_failed(task_id: str, error: str) -> None:
            if task_id != self.output_preview_task_id:
                return
            self.output_status.setText(f"preview failed: {error}")
            self.log(f"output preview failed: {error}")

        self.output_preview_task_id = _start_output_task(self, "Loading photo preview...", fetch_photo, photo_ready, photo_failed)
        return

    self.show_video_output(item)


def show_video_output(self: base.MainWindow, item: dict[str, Any]) -> None:
    _ensure_output_worker_state(self)
    path = str(item.get("download_path") or "")
    relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
    safe_relative = relative.replace("/", "_").replace("\\", "_")
    local_name = f"{item.get('source', 'output')}_{safe_relative}"
    local_path = self.output_temp_dir / local_name
    self.output_preview.setPixmap(QPixmap())
    self.output_preview.setText(f"Loading video preview:\n{relative}")

    def fetch_video() -> dict[str, str]:
        if not local_path.exists() or local_path.stat().st_size != int(item.get("size") or -1):
            self.client.download_file(path, local_path, timeout=900.0)
        return {"relative": relative, "local_path": str(local_path)}

    def video_ready(task_id: str, result: object) -> None:
        if task_id != self.output_preview_task_id:
            return
        payload = result if isinstance(result, dict) else {}
        ready_relative = str(payload.get("relative") or relative)
        ready_path = Path(str(payload.get("local_path") or local_path))
        if self.output_player is None or self.output_video is None:
            self.output_preview.show()
            self.output_preview.setText(
                f"Video ready to download:\n{ready_relative}\n\nInstall PySide6 multimedia support for inline playback."
            )
            self.output_status.setText(f"Selected video {ready_relative}")
            return
        self.output_preview.hide()
        self.output_video.show()
        self.output_player.setSource(QUrl.fromLocalFile(str(ready_path)))
        self.output_player.play()
        self.output_status.setText(f"Playing {ready_relative}")

    def video_failed(task_id: str, error: str) -> None:
        if task_id != self.output_preview_task_id:
            return
        self.output_status.setText(f"preview failed: {error}")
        self.log(f"output preview failed: {error}")

    self.output_preview_task_id = _start_output_task(self, f"Loading video preview: {relative}", fetch_video, video_ready, video_failed)


def download_current_output(self: base.MainWindow) -> None:
    item = self.current_output()
    if not item:
        self.output_status.setText("No output selected")
        return
    folder = QFileDialog.getExistingDirectory(self, "Download selected output to folder")
    if not folder:
        return
    item = dict(item)
    destination = Path(folder) / str(item.get("name") or Path(str(item.get("relative_path"))).name)

    def download() -> str:
        return str(self.client.download_file(str(item.get("download_path")), destination))

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"Downloaded to {result}")
        self.log(f"downloaded output: {result}")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"download failed: {error}")
        self.log(f"download failed: {error}")

    self.output_download_task_id = _start_output_task(self, f"Downloading {destination.name}...", download, succeeded, failed)


def download_all_outputs(self: base.MainWindow) -> None:
    if not self.output_files:
        self.output_status.setText("No outputs to download")
        return
    folder = QFileDialog.getExistingDirectory(self, "Download all listed outputs to folder")
    if not folder:
        return
    destination_root = Path(folder)
    files = [dict(item) for item in self.output_files]

    def download_all() -> str:
        for item in files:
            relative = Path(str(item.get("relative_path") or item.get("name")))
            destination = destination_root / str(item.get("source") or "output") / relative
            self.client.download_file(str(item.get("download_path")), destination)
        return str(destination_root)

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"Downloaded {len(files)} file(s) to {result}")
        self.log(f"downloaded {len(files)} output file(s) to {result}")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"download all failed: {error}")
        self.log(f"download all failed: {error}")

    self.output_download_task_id = _start_output_task(self, f"Downloading {len(files)} file(s)...", download_all, succeeded, failed)


def check_connection(self: base.MainWindow) -> None:
    self.sync_settings()
    self.tabs.setCurrentWidget(self.log_box)
    _ensure_output_worker_state(self)
    self.log("checking connection...")

    def fetch_health() -> dict[str, Any]:
        return self.client.request_json("GET", "/health", timeout=5.0)

    def succeeded(task_id: str, payload: object) -> None:
        if task_id != self.output_health_task_id:
            return
        self.log("health: " + base.json.dumps(payload, indent=2))

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_health_task_id:
            return
        self.log(f"health failed: {error}")

    self.output_health_task_id = _start_output_task(self, "Checking connection...", fetch_health, succeeded, failed)


def install() -> None:
    base.MainWindow.check_connection = check_connection
    base.MainWindow.refresh_outputs = refresh_outputs
    base.MainWindow.show_output_at = show_output_at
    base.MainWindow.show_video_output = show_video_output
    base.MainWindow.download_current_output = download_current_output
    base.MainWindow.download_all_outputs = download_all_outputs


install()
main = base.main