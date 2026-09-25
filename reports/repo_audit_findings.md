# Repository Audit Findings

_Audit performed 2026-09-25. Filesystem timestamps use `+05:30`._

## Executive Findings

- The VGG16 segmentation Dice discrepancy is unresolved by provenance evidence.
- `0.9043` appears only in `research/model_comparison.md` as a terminal-only result.
- `0.9223` is recorded in `reports/eval_unet.json` for `unet_vgg16`, with evaluation timestamp `2026-09-20T18:25:52.828100`.
- The available files do not establish whether the values came from different checkpoints, metric aggregation, thresholds, or validation protocols.
- `models/unet_mobilenetv2.pth` has now been evaluated. `reports/eval_unet_mobilenetv2.json` records Dice=0.8904 and Jaccard=0.8206 on the comparable validation split.
- Edge deployment remains unmeasured. `edge/benchmark_results.md` reports no FPS or latency values.

## VGG16 Dice Evidence

| Source | Value | Evidence | Last modified |
|---|---:|---|---|
| `research/model_comparison.md` | `0.9043` | Claims Kaggle T4, 30 epochs plus 15 fine-tune epochs; no saved metric artifact | `2026-09-21T01:14:26.7569789+05:30` |
| `reports/eval_unet.json` | `0.9223` | `model=unet_vgg16`, 389 validation samples, `val_fraction=0.15`, `seed=42`, `image_size=256` | `2026-09-20T23:56:31.1439051+05:30` |
| `reports/final_recommendation.md` | `0.9223` | References the saved evaluation result | `2026-09-21T01:16:37.0849820+05:30` |

The repository does not prove that `0.9043` is an error, a retrain result, or a different metric protocol. The later JSON-backed `0.9223` is better supported as the current reported value, but the underlying run relationship is unverified.

## Dataset Verification

Fresh filesystem counts:

| Dataset | Documented count | Count on disk | Result |
|---|---:|---:|---|
| HAM10000 images | 10,015 | 10,015 | Matches |
| ISIC 2018 images | 2,594 | 2,594 | Matches |
| ISIC 2018 masks | 2,594 | 2,594 | Matches |
| ISIC 2019 full dataset | 50,662 | 25,331 input images | Only half is downloaded locally |
| ISIC 2024 training images | 401,059 | 401,059 | Matches |

## Literature Review

The synthesis section remains present in `research/literature_review.md` and recommends:

1. EfficientNet-B0 for on-device classification.
2. EfficientFormerV2 or small Swin Transformer for research benchmarking.
3. U-Net with VGG16 for segmentation, with MobileNetV2 as a proposed edge swap.

## MobileNetV2 Segmentation

- Checkpoint: `models/unet_mobilenetv2.pth`
- Size: `35,747,747` bytes
- Last modified: `2026-09-21T23:22:58.0403839+05:30`
- Evaluation JSON: `reports/eval_unet_mobilenetv2.json`
- Dice: `0.8904298734664917` (`0.8904` rounded)
- Jaccard: `0.8205638313293457` (`0.8206` rounded)
- Training completion: checkpoint evaluated successfully; training provenance remains unverified

Because Dice is below `0.90`, VGG16 remains selected and MobileNetV2 is evaluated-but-not-adopted.

## Edge Deployment

Current `edge/` files:

| File | Size | Last modified |
|---|---:|---|
| `benchmark_jetson.py` | 4,311 bytes | `2026-09-21T00:51:11.7534126+05:30` |
| `benchmark_results.md` | 2,072 bytes | `2026-09-21T00:52:35.3902448+05:30` |
| `efficientnet_b0_ham10000.onnx` | 654,481 bytes | `2026-09-21T00:52:20.2513470+05:30` |
| `efficientnet_b0_ham10000.onnx.data` | 16,056,320 bytes | `2026-09-21T00:52:20.2292779+05:30` |
| `export_to_onnx.py` | 4,284 bytes | `2026-09-21T00:50:56.3592845+05:30` |

No verified Jetson Nano FPS, latency, memory, FP16, or INT8 measurements exist.

## Model Checkpoint Inventory

| File | Size | Last modified |
|---|---:|---|
| `experiments/efficientnet_b0_ham10000_focal.pth` | 16,373,265 bytes | `2026-09-19T15:51:31.4618518+05:30` |
| `models/efficientnet_b0_ham10000.pth` | 16,370,941 bytes | `2026-09-18T19:51:13.8707996+05:30` |
| `models/efficientnet_b0_isic2019.pth` | 16,376,061 bytes | `2026-09-18T23:56:31.2651230+05:30` |
| `models/swin_small_ham10000.pth` | 195,961,489 bytes | `2026-09-19T11:59:09.8706531+05:30` |
| `models/swin_small_ham10000_v2.pth` | 195,963,590 bytes | `2026-09-20T23:38:25.8244266+05:30` |
| `models/unet_mobilenetv2.pth` | 35,747,747 bytes | `2026-09-21T23:22:58.0403839+05:30` |
| `models/unet_vgg16.pth` | 167,627,979 bytes | `2026-09-18T18:29:17.1360766+05:30` |

## Evaluation JSON Inventory

| File | Size | Last modified |
|---|---:|---|
| `reports/eval_ce_baseline.json` | 5,151 bytes | `2026-09-19T18:07:44.4491876+05:30` |
| `reports/eval_focal.json` | 553 bytes | `2026-09-19T16:01:14.9773475+05:30` |
| `reports/eval_ham10000_20260919_180000.json` | 5,180 bytes | `2026-09-19T18:10:24.9714917+05:30` |
| `reports/eval_swin_v2.json` | 1,744 bytes | `2026-09-20T23:50:43.1522076+05:30` |
| `reports/eval_unet.json` | 280 bytes | `2026-09-20T23:56:31.1439051+05:30` |
| `reports/eval_unet_mobilenetv2.json` | 395 bytes | `2026-09-25T23:23:58.529100+05:30` |

MobileNetV2 evaluation is complete; its result is below the deployment threshold.
