# Browser-Based Real-Time Face Swap with Colab GPU

A simplified approach to real-time face swapping that runs entirely in a Google Colab notebook - no Windows client needed!

## How It Works

```
Browser (Webcam Capture)
    ↓ JavaScript captures frames
    ↓ Base64 encoding
Google Colab Notebook
    ↓ Python receives frames
    ↓ GPU processes (InsightFace)
    ↓ Returns processed frames
Browser (Display Result)
```

## Features

✅ **No installation** - Runs entirely in browser + Colab
✅ **GPU acceleration** - Uses Colab's free T4 GPU
✅ **Privacy-first** - Webcam data stays in your session
✅ **Simple setup** - Just upload a source face image and go
✅ **No networking** - No ZMQ, Tailscale, or ffmpeg complexity

## Usage

1. **Open the notebook in Google Colab**:
   - Upload `Browser_Webcam_Face_Swap.ipynb` to Google Colab
   - Or use this link: [Open in Colab](https://colab.research.google.com)

2. **Select GPU runtime**:
   - Runtime → Change runtime type → T4 GPU

3. **Run cells in order**:
   - Install dependencies
   - Verify GPU
   - Download face swap model
   - Upload your source face image
   - Start webcam capture

4. **Allow webcam access** when prompted

5. **View real-time face swap** in the notebook output!

## Performance

- **FPS**: 2-5 FPS (limited by browser → Colab → browser round-trip)
- **Latency**: ~200-500ms per frame
- **GPU**: T4 GPU processes each frame in ~50-100ms
- **Resolution**: 640×480 (configurable)

## Limitations

❌ **Not for production** - This is a demo/preview tool
❌ **No virtual camera** - Output stays in notebook (can't use in Zoom/Teams)
❌ **Lower FPS** - Browser API overhead limits frame rate
❌ **Requires Colab session** - Must stay connected to Colab

## Comparison to Windows Client Version

| Feature | Browser-Based (This) | Windows Client |
|---------|---------------------|----------------|
| Installation | None | Requires Python, dependencies |
| Performance | 2-5 FPS | 15-30 FPS |
| Virtual Camera | ❌ No | ✅ Yes (OBS) |
| Zoom/Teams | ❌ No | ✅ Yes |
| Setup Complexity | Low | High |
| Network Requirements | None | Tailscale/ZMQ |

## When to Use This

✅ **Quick testing** - Test face swap models quickly
✅ **Demos** - Show face swapping in presentations
✅ **Learning** - Understand how face swapping works
✅ **No installation** - Don't want to set up Windows client

## When to Use Windows Client Instead

✅ **Production use** - Need reliable high FPS
✅ **Video calls** - Use in Zoom/Discord/Teams
✅ **Recording** - Need to record face-swapped video
✅ **Lower latency** - Need <100ms latency

## Architecture

Based on the modern Deep-Live-Cam upstream (from `tmp/Deep-Live-Cam`) with browser-based webcam capture added.

Uses:
- **InsightFace** - Face detection and recognition
- **ONNX Runtime GPU** - Fast inference on T4 GPU
- **JavaScript MediaDevices API** - Browser webcam capture
- **Colab eval_js** - Python ↔ JavaScript communication

## Credits

Based on:
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)
- [InsightFace](https://github.com/deepinsight/insightface)

## License

Same as Deep-Live-Cam (check upstream repository)
