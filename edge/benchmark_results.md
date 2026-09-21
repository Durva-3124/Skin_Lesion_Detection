# Module 5 — Edge Deployment Benchmark Results

## Conversion Status

| Step | Status | Output | Notes |
|---|---|---|---|
| ONNX export | **COMPLETE** | `edge/efficientnet_b0_ham10000.onnx` (0.7 MB) | Exported on 2025-07-14, opset 18, input (1,3,224,224), verified with `onnx.checker.check_model()` |
| TensorRT FP16 build | **PENDING — requires JetPack** | `edge/efficientnet_b0_ham10000_fp16.trt` | Script written and ready: `edge/export_to_onnx.py --build-trt`. Requires `tensorrt` + `pycuda` from JetPack 5.x. Not runnable on this development machine. |
| On-device benchmark | **PENDING — requires hardware** | — | Script written and ready: `edge/benchmark_jetson.py`. Measures mean/P50/P95/P99 latency and FPS over 200 iterations with 20 warmup passes. |

## On-Device Performance

**NOT TESTED — Jetson Nano hardware not available.**

No FPS or latency numbers are reported here. The benchmarking script (`edge/benchmark_jetson.py`) is fully written and will append measured results to this file when run on-device.

Literature estimates for EfficientNet-B0 on Jetson Nano (Maxwell GPU, TensorRT FP16): 15–25 FPS. These are **estimates only** — do not present them as measured results. Real numbers must come from `edge/benchmark_jetson.py` running on the actual device.

## Export Details

- Source model: `models/efficientnet_b0_ham10000.pth`
- Architecture: EfficientNet-B0, 7-class HAM10000 classifier
- ONNX opset: 18 (torch 2.12.1 default; opset 17 requested but version converter fell back to 18 — this is compatible with TensorRT 8.x+)
- Input shape: fixed (1, 3, 224, 224), batch=1 for Jetson Nano deployment
- Precision target: FP16 (TensorRT `BuilderFlag.FP16`)
- Workspace: 1 GB

## How to Complete Module 5

On a JetPack 5.x environment (Jetson Nano or compatible):

```bash
# Install dependencies
sudo apt-get install python3-libnvinfer-dev
pip install pycuda

# Build TensorRT engine
python edge/export_to_onnx.py --build-trt

# Run benchmark
python edge/benchmark_jetson.py
```

Results will be appended to this file automatically.
