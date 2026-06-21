#!/usr/bin/env python3
"""Test the exact sequence that Deep-Live-Cam uses"""

import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat

print("Step 1: Open physical webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
if not cap.isOpened():
    print("❌ Failed to open webcam")
    exit(1)
print("✓ Webcam opened")

print("\nStep 2: Test reading a frame...")
ok, frame = cap.read()
if not ok:
    print("❌ Failed to read frame")
    cap.release()
    exit(1)
print(f"✓ Read frame: {frame.shape}")

print("\nStep 3: Try to create virtual camera (NO device specified)...")
try:
    with pyvirtualcam.Camera(
        width=480,
        height=240,
        fps=15,
        fmt=PixelFormat.BGR,
        print_fps=False,
    ) as virtual_cam:
        print(f"✓ Virtual camera created: {virtual_cam.device} (backend: {virtual_cam.backend})")

        print("\nStep 4: Send one frame to virtual camera...")
        if frame.shape[1] != 480 or frame.shape[0] != 240:
            frame = cv2.resize(frame, (480, 240))
        virtual_cam.send(frame)
        print("✓ Frame sent successfully")

        print("\n✅ SUCCESS! Virtual camera works with this sequence.")

except Exception as e:
    print(f"\n❌ FAILED at virtual camera creation: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    print("\n✓ Webcam released")
