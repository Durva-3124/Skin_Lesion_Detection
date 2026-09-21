"""
edge/benchmark_jetson.py
On-device inference benchmark for TejaLens on Jetson Nano.
Measures FPS and latency for the TensorRT FP16 engine.

Run on Jetson Nano only:
    python edge/benchmark_jetson.py

Requirements (JetPack environment):
    tensorrt, pycuda, numpy, Pillow, opencv-python

Outputs to stdout and appends results to edge/benchmark_results.md.
"""

import pathlib
import time
import datetime
import numpy as np

TRT_PATH   = pathlib.Path(__file__).parent / "efficientnet_b0_ham10000_fp16.trt"
WARMUP     = 20    # warmup iterations (not measured)
ITERATIONS = 200   # measured iterations
IMAGE_SIZE = 224


def load_engine(trt_path: pathlib.Path):
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(str(trt_path), "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    import pycuda.driver as cuda
    import tensorrt as trt

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    return inputs, outputs, bindings, stream


def run_inference(context, inputs, outputs, bindings, stream):
    import pycuda.driver as cuda

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    return outputs[0]["host"]


def benchmark():
    if not TRT_PATH.exists():
        print(f"ERROR: TRT engine not found at {TRT_PATH}")
        print("Run: python edge/export_to_onnx.py --export-onnx --build-trt")
        return

    print(f"Loading TRT engine: {TRT_PATH}")
    engine = load_engine(TRT_PATH)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Synthetic input (ImageNet-normalized random image)
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    np.copyto(inputs[0]["host"], dummy.ravel())

    print(f"Warming up ({WARMUP} iterations)...")
    for _ in range(WARMUP):
        run_inference(context, inputs, outputs, bindings, stream)

    print(f"Benchmarking ({ITERATIONS} iterations)...")
    latencies = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        run_inference(context, inputs, outputs, bindings, stream)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    mean_ms  = float(latencies.mean())
    p50_ms   = float(np.percentile(latencies, 50))
    p95_ms   = float(np.percentile(latencies, 95))
    p99_ms   = float(np.percentile(latencies, 99))
    fps      = 1000.0 / mean_ms

    print(f"\n--- Benchmark Results ---")
    print(f"Mean latency : {mean_ms:.2f} ms")
    print(f"P50 latency  : {p50_ms:.2f} ms")
    print(f"P95 latency  : {p95_ms:.2f} ms")
    print(f"P99 latency  : {p99_ms:.2f} ms")
    print(f"FPS          : {fps:.1f}")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    result_lines = [
        "",
        f"## On-Device Run: {timestamp}",
        f"- Engine: {TRT_PATH.name}",
        f"- Precision: FP16",
        f"- Input: {IMAGE_SIZE}x{IMAGE_SIZE}, batch=1",
        f"- Warmup: {WARMUP} | Measured: {ITERATIONS}",
        f"- Mean latency: {mean_ms:.2f} ms | FPS: {fps:.1f}",
        f"- P50: {p50_ms:.2f} ms | P95: {p95_ms:.2f} ms | P99: {p99_ms:.2f} ms",
    ]
    results_path = pathlib.Path(__file__).parent / "benchmark_results.md"
    with open(str(results_path), "a", encoding="utf-8") as f:
        f.write("\n".join(result_lines) + "\n")
    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    benchmark()
