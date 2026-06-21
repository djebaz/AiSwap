#!/usr/bin/env python3
"""Test ZMQ connection to Colab server"""

import zmq
import sys

REMOTE_HOST = "100.124.216.103"
PORT = 5555
TIMEOUT_MS = 5000

print(f"Testing ZMQ connection to {REMOTE_HOST}:{PORT}...")
print(f"Timeout: {TIMEOUT_MS}ms")

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
socket.setsockopt(zmq.SNDTIMEO, TIMEOUT_MS)

try:
    address = f"tcp://{REMOTE_HOST}:{PORT}"
    print(f"\nConnecting to {address}...")
    socket.connect(address)
    print("✓ Socket connected")

    print("\nSending test multipart message (mimicking real protocol)...")
    import json
    import numpy as np

    # Create a tiny test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_bytes = test_image.tobytes()

    # Create metadata matching the expected format
    metadata = {
        'manyface': False,
        'enhance': False,
        'dtype_source': str(test_image.dtype),
        'shape_source': test_image.shape,
        'size': '640x480',
        'fps': '60'
    }
    metadata_json = json.dumps(metadata).encode('utf-8')

    # Send as multipart message (metadata + bytes)
    socket.send_multipart([metadata_json, test_bytes])
    print(f"✓ Sent multipart message ({len(metadata_json)} + {len(test_bytes)} bytes)")

    print("\nWaiting for ACK response...")
    response = socket.recv_string()
    print(f"✓ Received response: {response}")

    print("\n✅ SUCCESS! ZMQ connection is working!")
    sys.exit(0)

except zmq.Again as e:
    print(f"\n❌ TIMEOUT: No response within {TIMEOUT_MS}ms")
    print("This means the server is not responding on port 5555")
    print("\nPossible causes:")
    print("  1. Colab LIVE_SERVER is not running (check Cell 12 output)")
    print("  2. Tailscale serve is not configured (check Cell 15 output)")
    print("  3. Colab server crashed (check Cell 16 health check)")
    sys.exit(1)

except zmq.ZMQError as e:
    print(f"\n❌ ZMQ ERROR: {e}")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    socket.close()
    context.term()
