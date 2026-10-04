"""
edge/export_to_onnx.py
Export efficientnet_b0_ham10000.pth to ONNX, then convert to TensorRT FP16 engine.

Usage:
    # Step 1: ONNX export (runs on any machine with PyTorch)
    python edge/export_to_onnx.py --export-onnx

    # Step 2: TensorRT conversion (requires TensorRT + JetPack 4.6.x environment)
    python edge/export_to_onnx.py --build-trt

    # Both steps:
    python edge/export_to_onnx.py --export-onnx --build-trt

Outputs:
    edge/efficientnet_b0_ham10000.onnx
    edge/efficientnet_b0_ham10000_fp16.trt

Note on JetPack: The original Jetson Nano supports JetPack 4.6.x (TensorRT 8.0.x).
JetPack 5.x is NOT supported on the original Jetson Nano (Maxwell GPU).
Verify your JetPack version before running --build-trt.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch

ROOT        = pathlib.Path(__file__).parent.parent
CLS_WEIGHTS = ROOT / "models" / "efficientnet_b0_ham10000.pth"
EDGE_DIR    = ROOT / "edge"
ONNX_PATH   = EDGE_DIR / "efficientnet_b0_ham10000.onnx"
TRT_PATH    = EDGE_DIR / "efficientnet_b0_ham10000_fp16.trt"

BATCH_SIZE  = 1
IMAGE_SIZE  = 224
NUM_CLASSES = 7
DEFAULT_OPSET = 13  # opset 13 for TensorRT 8.x compatibility; use --opset 18 for newer runtimes


def export_onnx(opset: int = DEFAULT_OPSET):
    """Export EfficientNet-B0 to ONNX with fixed input shape (1, 3, 224, 224)."""
    from src.classification.model import get_classification_model

    print("Loading model...")
    model = get_classification_model("efficientnet_b0", num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(str(CLS_WEIGHTS), map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    print(f"Exporting to ONNX (opset {opset}): {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed batch=1 for Jetson Nano deployment
    )
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"ONNX export complete: {ONNX_PATH} ({size_mb:.1f} MB)")
    if ONNX_PATH.with_suffix(".onnx.data").exists():
        print("WARNING: external .onnx.data file created — weights not embedded. "
              "Consider using opset 13 or torch.onnx.export with keep_initializers_as_inputs=False.")

    # Verify ONNX model is valid
    import onnx
    model_onnx = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed.")

    # Verify output matches PyTorch on 20 test images
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        max_diffs = []
        for _ in range(20):
            x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            with torch.no_grad():
                pt_out = model(x).numpy()
            ort_out = sess.run(None, {input_name: x.numpy()})[0]
            max_diffs.append(float(abs(pt_out - ort_out).max()))
        print(f"PyTorch vs ONNX max abs diff over 20 samples: {max(max_diffs):.2e}")
    except ImportError:
        print("onnxruntime not installed — skipping output verification. Run: pip install onnxruntime")


def build_trt():
    """
    Convert ONNX to TensorRT FP16 engine using trtexec or the Python TensorRT API.
    Requires: tensorrt, pycuda (JetPack environment).
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt not installed. Run this step on a JetPack-compatible environment.")
        print("Install: sudo apt-get install python3-libnvinfer-dev (JetPack 5.x)")
        sys.exit(1)

    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX file not found at {ONNX_PATH}. Run --export-onnx first.")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(str(ONNX_PATH), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"TRT parse error: {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled.")
    else:
        print("WARNING: FP16 not supported on this platform, falling back to FP32.")
    # Note: Jetson Nano (Maxwell GPU, JetPack 4.6.x) has limited FP16 support.
    # Verify with: python -c "import tensorrt as trt; b=trt.Builder(trt.Logger()); print(b.platform_has_fast_fp16)"

    print(f"Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: TensorRT engine build failed.")
        sys.exit(1)

    with open(str(TRT_PATH), "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved: {TRT_PATH} ({TRT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--build-trt",   action="store_true")
    parser.add_argument("--opset",        type=int, default=DEFAULT_OPSET,
                        help=f"ONNX opset version (default: {DEFAULT_OPSET})")
    args = parser.parse_args()

    if not args.export_onnx and not args.build_trt:
        parser.print_help()
        sys.exit(0)

    EDGE_DIR.mkdir(parents=True, exist_ok=True)

    if args.export_onnx:
        export_onnx(opset=getattr(args, "opset", DEFAULT_OPSET))
    if args.build_trt:
        build_trt()
