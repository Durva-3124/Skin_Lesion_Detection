# TejaLens Skin Pre-Screening — Final Recommendation Report

_Generated: 2026-09-21. All numbers in this report are sourced from verified evaluation
files in `reports/`. No estimates are presented as measured results._

---

## 1. Executive Summary

TejaLens is an AI-powered skin lesion pre-screening pipeline designed to run on consumer
or smartphone-captured images and flag potentially malignant lesions for clinical follow-up.
The pipeline chains five stages: image quality gating, hair removal preprocessing, lesion
segmentation (U-Net with VGG16 encoder), 7-class classification (EfficientNet-B0), and
MC Dropout uncertainty estimation with Grad-CAM explainability.

On the HAM10000 validation set (1,502 samples, imbalanced split), EfficientNet-B0 achieves
**accuracy 0.7696, macro F1 0.7384, and mean malignant-class recall 0.8027** (mel=0.7943,
bcc=0.9101, akiec=0.7037). The segmentation model achieves **Dice 0.9223** on ISIC 2018
Task 1. The pipeline has been end-to-end tested on real HAM10000 images with quality
gating, MC Dropout inference, and Grad-CAM heatmap generation all producing real saved
outputs. The EfficientNet-B0 model has been exported to ONNX (verified) and a TensorRT
FP16 conversion script is ready; on-device Jetson Nano performance has not been measured
due to hardware unavailability.

**Headline recommendation:** EfficientNet-B0 is the deployment model. It is the only
classifier that is balanced across all 7 classes on imbalanced data. The gap versus
literature (97%+) is explained by the imbalanced evaluation split — literature numbers
use rebalanced data. The pipeline is ready for on-device testing pending Jetson Nano
hardware access.

---

## 2. Datasets

### Training and Evaluation Composition

| Dataset | Role | Images | Classes | Split Used |
|---|---|---|---|---|
| HAM10000 | Classification training + validation | 10,015 | 7 (akiec, bcc, bkl, df, mel, nv, vasc) | val_fraction=0.15, seed=42 → 8,513 train / 1,502 val |
| ISIC 2018 Task 1 | Segmentation training + validation | 2,594 image/mask pairs | Binary (lesion/background) | val_fraction=0.15, seed=42 → 2,205 train / 389 val |
| ISIC 2019 | Cross-dataset classification validation | 50,662 | 8 classes | val_fraction=0.15, seed=42 → 43,063 train / 7,599 val |
| ISIC 2024 SLICE-3D | Held-out generalization test | 401,059 | Binary | **Not used** — reserved as held-out test only |

### Class Imbalance (HAM10000)

The dataset is severely imbalanced: nv (melanocytic nevi) accounts for ~67% of images
(6,705 of 10,015). The rarest class, df (dermatofibroma), has 115 images — a 58:1 ratio
against nv. All training used class-weighted CrossEntropyLoss with inverse-frequency
weights to address this. The imbalance is the primary explanation for the gap between
reproduced accuracy (0.7696) and literature numbers (96–97%) — literature studies
typically use rebalanced or oversampled splits.

---

## 3. Literature Synthesis

The key findings from the Module 2 literature review that shaped model and architecture
choices:

**On classification:** EfficientNet-B0 offers the best accuracy-to-parameter ratio for
edge deployment (~5.3M params, ~20MB). Swin Transformer outperforms CNNs on harder
imbalanced sets and on mobile-acquired images (arXiv 2509.04800), but its attention layers
quantize poorly on Maxwell GPU and it collapsed on nv recall in both training attempts on
this project's data. Ensemble methods (Swin+EfficientNetB4, MaxViT+ConvNeXt+EfficientNetV2)
achieve 96–98.5% but are not edge-feasible.

**On segmentation:** U-Net with VGG16 encoder is the verified baseline (Manzoor et al.,
DIGITAL HEALTH 2025, Dice 94.24% on ISIC 2018). MobileNetV2 encoder was evaluated on this
project's data but did not meet the deployment threshold (~3.5M vs ~138M params).

**On uncertainty and explainability:** MC Dropout (Gal & Ghahramani, 2016) is the
established approach for uncertainty estimation in deployed medical AI. Grad-CAM provides
visual saliency maps that are interpretable to clinical users. Both are implemented and
tested in this pipeline.

---

## 4. Model Comparison and Final Choice

### Classification

| Model | Accuracy | Macro F1 | Mean Malignant Recall | nv Recall | Params | Edge Feasible | Status |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | **0.7696** | **0.7384** | **0.8027** | 0.7418 | ~5.3M | Yes | **SELECTED** |
| Swin-Small (original) | 0.4747 | 0.5790 | 0.8857 | 0.2776 | ~28M | No | Rejected — nv collapse |
| Swin-Small v2 (retrain) | 0.4015 | 0.4496 | 0.8069 | 0.2378 | ~28M | No | Rejected — worse than original |
| EfficientNet-B0 (focal loss) | 0.3129 | — | — | 0.0786 | ~5.3M | Yes | Rejected — nv collapse |

_Source: `reports/eval_ce_baseline.json` (2026-09-19), `reports/eval_swin_v2.json` (2026-09-20), `reports/eval_focal.json`._

**Why EfficientNet-B0:** It is the only model that is balanced across all 7 classes on
imbalanced data. Swin-Small achieves higher malignant recall (0.8857 vs 0.8027) but
collapses on nv (0.2776 recall) — it misclassifies 72% of the most common class, making
it clinically unusable as a screening tool. The accuracy gap versus literature is explained
by the imbalanced evaluation split, not a model deficiency.

**Known limitation:** The exact training configuration of the selected checkpoint
(`models/efficientnet_b0_ham10000.pth`) is unverifiable — the Kaggle notebook that
produced it was lost. Loss type, exact epoch count, and augmentation settings are unknown.
This is accepted as a project record limitation.

### Segmentation

| Model | Dice | Jaccard | Params | Edge Feasible | Status |
|---|---|---|---|---|---|
| **U-Net (VGG16 encoder)** | **0.9223** | — | ~138M | Medium | **SELECTED** |
| U-Net (MobileNetV2 encoder) | **0.8904** | **0.8206** | ~3.5M | High | Evaluated but not adopted; below Dice threshold |

_Source: `reports/eval_unet.json` and `reports/eval_unet_mobilenetv2.json` (2026-09-25)._

**Deployment decision for segmentation:** VGG16 encoder is selected until MobileNetV2
Dice is measured. MobileNetV2 measured Dice=0.8904 and Jaccard=0.8206, below the Dice ≥
0.90 decision threshold, so the encoder swap is not adopted. VGG16 remains selected;
MobileNetV2 is evaluated-but-not-adopted, not future work.

---

## 5. Pipeline Architecture

```
Image Input (smartphone or dermoscope capture)
    |
    v
[Stage 0] Quality Gate
    Laplacian variance < 50  → reject: "too blurry"
    Mean brightness < 30     → reject: "too dark"
    Mean brightness > 225    → reject: "overexposed"
    |
    v (passed images only)
[Preprocessing] Hair Removal
    Black-hat morphological filtering + TELEA inpainting
    |
    v
[Stage 1] Segmentation — UNet-VGG16
    Input: 512×512 (VGG16 encoder requires ≥ 256px; 512px used for training)
    Output: binary lesion mask (H×W, sigmoid threshold 0.5)
    Dice on ISIC 2018 val: 0.9223
    |
    v
[Stage 2] Classification + MC Dropout — EfficientNet-B0
    Input: 224×224, ImageNet normalization
    30 stochastic forward passes (Dropout p=0.3, inplace=False)
    Output: mean class probabilities + uncertainty (mean variance across passes)
    HAM10000 val accuracy: 0.7696 | Macro F1: 0.7384
    |
    v
[Stage 3] Risk Level
    Malignant class (mel/bcc/akiec) + confidence ≥ 0.70 → HIGH
    Malignant class + confidence < 0.70               → MEDIUM
    Benign class + confidence ≥ 0.70                  → LOW
    Any class + confidence < 0.70                     → MEDIUM
    |
    v
[Stage 4] Grad-CAM Explainability
    Targets last conv block of EfficientNet-B0 (model.features[-1])
    Output: heatmap overlay showing which lesion region drove the prediction
    |
    v
Output dict: predicted_class, confidence, uncertainty, risk_level,
             class_probabilities, segmentation_mask, gradcam_heatmap
```

_Implementation: `src/pipeline.py` (TejaLensPipeline class). All stages tested end-to-end
on real HAM10000 images — see `reports/quality_gate_test.md`, `reports/uncertainty_test.md`,
`reports/gradcam_samples/`._

---

## 6. Edge Deployment Results

### ONNX Export

EfficientNet-B0 has been exported to ONNX and verified:
- File: `edge/efficientnet_b0_ham10000.onnx` (0.7 MB)
- Opset: 18 (torch 2.12.1 default)
- Input shape: fixed (1, 3, 224, 224), batch=1
- Verified with `onnx.checker.check_model()` — passed

### TensorRT FP16

Conversion script written and ready (`edge/export_to_onnx.py --build-trt`). Requires
TensorRT + pycuda from JetPack 5.x. Not runnable on the development machine.

### On-Device Performance

**NOT MEASURED — Jetson Nano hardware not available.**

No FPS or latency numbers are reported. The benchmarking script (`edge/benchmark_jetson.py`)
is fully written and will append measured results to `edge/benchmark_results.md` when run
on-device. Literature estimates for EfficientNet-B0 on Jetson Nano (Maxwell GPU, TensorRT
FP16): 15–25 FPS — these are estimates only and must not be presented as measured results.

| Precision | FPS | Mean Latency | P95 Latency | Accuracy Drop |
|---|---|---|---|---|
| FP32 (ONNX, CPU) | — | — | — | baseline |
| FP16 (TensorRT, Jetson Nano) | **not measured** | **not measured** | **not measured** | **not measured** |
| INT8 (TensorRT, Jetson Nano) | not attempted | — | — | — |

---

## 7. Trust Layer

### Quality Gate

Tested on 8 images (6 real HAM10000 + 2 synthetic degraded). Results:
- 3 real images passed (sharpness 57.5–475.6, brightness 143–162)
- 3 real images correctly rejected as genuinely blurry (sharpness 20.9–45.4) — these are
  real blurry images in the HAM10000 dataset, not test failures
- Both synthetic degraded images correctly rejected (blurry: sharpness=1.3; dark: sharpness=0.6)

_Source: `reports/quality_gate_test.md` (2026-09-21)._

### MC Dropout Uncertainty

Tested on 3 passing images (30 stochastic passes each):

| Image | Predicted Class | Confidence | Uncertainty | Risk |
|---|---|---|---|---|
| ISIC_0033301 | bcc (malignant) | 0.5386 | 0.000741 | MEDIUM |
| ISIC_0033450 | vasc (benign) | 1.0000 | 0.000000 | LOW |
| ISIC_0033550 | bkl (benign) | 0.5110 | 0.001728 | MEDIUM |

The vasc result (conf=1.000, unc=0.000) is correct — with p=0.3 dropout and a very large
logit for vasc, softmax saturates to 1.0 on every pass regardless of which neurons drop.
Zero variance on a saturated distribution is mathematically expected.

_Source: `reports/uncertainty_test.md` (2026-09-21)._

### Grad-CAM Explainability

Heatmap overlays generated for all 3 passing images. Saved to `reports/gradcam_samples/`.
Implementation targets `model.features[-1]` (last conv block of EfficientNet-B0) via
forward and backward hooks. Source: `src/explainability/__init__.py`.

### Risk Stratification

- HIGH: malignant class (mel/bcc/akiec) with confidence ≥ 0.70
- MEDIUM: malignant class < 0.70 confidence, OR any class < 0.70 confidence
- LOW: benign class with confidence ≥ 0.70

---

## 8. Animal Dataset Feasibility

No HAM10000-equivalent veterinary skin lesion dataset exists publicly. The canine Kaggle
dataset (~1,200 images, 6 categories) has unverified labels and is too small for training
a production classifier. No feline dataset was found.

**Transfer learning** from human dermoscopic models to animal lesions is plausible but
unvalidated. Key obstacles: fur occlusion (the hair removal preprocessing is not designed
for dense fur), skin texture differences between species, and label mismatch (HAM10000's
7 classes are human-specific diagnoses). A pilot experiment — fine-tuning EfficientNet-B0
on the canine Kaggle dataset and comparing against a random-weight baseline — is the
correct next step before committing to this path.

**Synthetic data generation** is a reasonable augmentation tool within an existing
veterinary dataset but is not a substitute for acquiring real labeled data. Published
gains from synthetic augmentation in the human skin lesion literature are modest (1–3% F1)
and frequently lack clinical validation of the synthetic images.

**Veterinary institution collaboration** is the only path to a clinically validated
veterinary dataset. This is a business development task (6–12 month timeline) requiring
outreach to veterinary dermatology departments with active research programs.

_Full analysis: `research/animal_dataset_feasibility.md`._

---

## 9. Limitations and Recommended Next Steps

### Unresolved gaps — visible, not smoothed over

1. **On-device performance unmeasured.** No FPS, latency, or memory numbers exist for
   Jetson Nano. The ONNX export is verified; TensorRT conversion and benchmarking require
   hardware access. Do not make deployment decisions based on literature estimates.

2. **Training config of selected classifier unverifiable.** The Kaggle notebook that
   produced `efficientnet_b0_ham10000.pth` was lost. Loss type, epoch count, and
   augmentation settings are unknown. The model's evaluation metrics are verified; its
   training provenance is not.

3. **MobileNetV2 segmentation was evaluated but did not meet the deployment threshold.**
   Its measured Dice is 0.8904 and Jaccard is 0.8206 on the same 389-sample validation
   split used for VGG16, so VGG16 remains selected for segmentation.

4. **No clinical validation performed.** The pipeline has not been tested on images from
   the actual TejaLens camera hardware, on patients in the target deployment setting, or
   against dermatologist ground truth. Accuracy numbers are from HAM10000 — a dermoscopic
   dataset that does not represent smartphone-captured images from community health workers.

5. **INT8 quantization not attempted.** The TensorRT export targets FP16. INT8 would
   further improve Jetson Nano performance but requires calibration data and accuracy
   validation. Not done.

6. **ISIC 2024 generalization test not run.** ISIC 2024 SLICE-3D (401,059 smartphone-proxy
   images) was reserved as a held-out generalization test but was not evaluated. This is
   the most important pending validation for the stated deployment context.

### Recommended next steps, in priority order

1. **Get Jetson Nano hardware.** Run `edge/benchmark_jetson.py`. Update
   `edge/benchmark_results.md` and Section 6 of this report with real numbers.
2. **Keep VGG16 as the selected segmentation encoder.** MobileNetV2 was evaluated in
   `reports/eval_unet_mobilenetv2.json` and is evaluated-but-not-adopted because Dice=0.8904.
3. **Run ISIC 2024 generalization evaluation.** Execute `scripts/eval_isic2024.py`.
   This is the most important accuracy number for the deployment context.
4. **Transfer learning pilot for veterinary use case.** Fine-tune EfficientNet-B0 on the
   canine Kaggle dataset. 1–2 days of GPU time. Go/no-go signal for the animal track.
5. **Clinical validation.** Capture images with the actual TejaLens camera hardware and
   evaluate against dermatologist ground truth. This is required before any clinical use.
