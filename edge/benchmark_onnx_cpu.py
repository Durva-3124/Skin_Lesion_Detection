"""
edge/benchmark_onnx_cpu.py
CPU latency benchmark for the exported ONNX classifier.
Reports mean/P50/P95/P99 latency over N iterations.
Results are CPU numbers — NOT Jetson Nano numbers.

Usage:
    python edge/benchmark_onnx_cpu.py
"""

import sys
import pathlib
import json
import datetime
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np

from src.config import REPORTS_DIR

EDGE_DIR   = pathlib.Path(__file__).parent
ONNX_PATH  = EDGE_DIR / "efficientnet_b0_ham10000.onnx"
WARMUP     = 20
ITERATIONS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",       default=str(ONNX_PATH))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--warmup",     type=int, default=WARMUP)
    args = parser.parse_args()

    onnx_path = pathlib.Path(args.onnx)
    if not onnx_path.exists():
        sys.exit(f"ERROR: ONNX file not found at {onnx_path}. Run: python edge/export_to_onnx.py --export-onnx")

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("ERROR: onnxruntime not installed. Run: pip install onnxruntime")

    import time

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print(f"ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Warming up ({args.warmup} passes)...")
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})

    print(f"Benchmarking ({args.iterations} passes)...")
    latencies = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    results = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "onnx_path":   str(onnx_path),
        "device":      "CPU (NOT Jetson Nano — do not compare with on-device numbers)",
        "iterations":  args.iterations,
        "warmup":      args.warmup,
        "latency_ms": {
            "mean": round(float(latencies.mean()), 2),
            "p50":  round(float(np.percentile(latencies, 50)), 2),
            "p95":  round(float(np.percentile(latencies, 95)), 2),
            "p99":  round(float(np.percentile(latencies, 99)), 2),
        },
        "throughput_fps": round(float(1000 / latencies.mean()), 1),
    }

    print(f"\nCPU Latency (ms): mean={results['latency_ms']['mean']} "
          f"P50={results['latency_ms']['p50']} "
          f"P95={results['latency_ms']['p95']} "
          f"P99={results['latency_ms']['p99']}")
    print(f"Throughput: {results['throughput_fps']} FPS (CPU only)")
    print("NOTE: These are CPU numbers, NOT Jetson Nano numbers.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "edge_cpu_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # Append summary to benchmark_results.md
    md_path = EDGE_DIR / "benchmark_results.md"
    with open(md_path, "a") as f:
        f.write(f"\n## CPU Benchmark — {results['timestamp']}\n")
        f.write(f"**Device: CPU (NOT Jetson Nano)**\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Mean latency | {results['latency_ms']['mean']} ms |\n")
        f.write(f"| P50 latency  | {results['latency_ms']['p50']} ms |\n")
        f.write(f"| P95 latency  | {results['latency_ms']['p95']} ms |\n")
        f.write(f"| P99 latency  | {results['latency_ms']['p99']} ms |\n")
        f.write(f"| Throughput   | {results['throughput_fps']} FPS |\n")
    print(f"Appended to {md_path}")


if __name__ == "__main__":
    main()
