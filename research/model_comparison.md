# Module 3 — Model Architecture Comparison
## TejaLens Skin Cancer Pre-Screening Module

_Numbers sourced from verified citations in `research/review.md` unless marked (literature estimate).
Reproduced benchmarks on Module 1 datasets added in Module 4 — marked **REPRODUCED**._

---

## The Core Tension (read this first)

Two accuracy targets exist in this project and they pull in opposite directions:

- **Research/benchmark model** — trained on GPU (Colab), evaluated on HAM10000/ISIC 2019. Optimize for maximum accuracy. This is what goes in the literature comparison table.
- **On-device model** — must run on Jetson Nano (Maxwell GPU, 128 CUDA cores, 4GB RAM). Must trade some accuracy for a model small and fast enough to run at usable frame rates after TensorRT INT8 quantization.

Do not pick one model for both. The tables below keep these two tracks separate.

---

## 1. Segmentation Model Comparison

All numbers on ISIC 2018 Task 1 (2,594 images + pixel-level masks).

| Model | Pixel Accuracy | Jaccard (IoU) | Dice | Params (approx.) | Jetson Nano Feasibility | Notes |
|---|---|---|---|---|---|---|
| **U-Net (VGG16 encoder)** | 97.59% | 89.12% | 94.24% | ~138M (VGG16 encoder) | Medium — encoder is large; swap to lighter encoder for on-device | VERIFIED: Manzoor et al., DIGITAL HEALTH 2025, DOI: 10.1177/20552076251351858. Standard baseline, best starting point. **REPRODUCED: Dice=0.9043 on ISIC 2018 Task 1 (2,594 images), Kaggle T4 GPU, 30 epochs + 15 fine-tune epochs.** |
| U-Net (MobileNetV2 encoder) | ~94–96% (literature estimate) | ~85–88% (literature estimate) | ~91–93% (literature estimate) | ~3.5M encoder | High — designed for edge, TensorRT-friendly | Not yet benchmarked on your data. Recommended swap for on-device deployment. Reproduce in Module 4. |
| GAN-assisted U-Net variants | ~95–97% | ~87–90% | ~92–95% | Heavier than baseline | Low — adversarial training cost, not edge-feasible | GAP in review.md — no single verified paper. Marginal gain over baseline for significantly higher training cost. |
| U-Net + Pyramid Vision Transformer (e.g. DBCGN) | Competitive with GAN variants | ~88–91% | ~93–95% | Large (PVT encoder) | Low — PVT too heavy for Jetson Nano without heavy pruning | GAP in review.md. Higher compute cost not justified for segmentation stage alone. |
| MRP-UNet (Res2-SE + pyramid dilated convolution) | ~96% | Not reported | ~94–96% | Medium | Medium | Scientific Reports 2025, DOI: 10.1038/s41598-025-92447-1. Attention/pyramid variant, not GAN-based. |

**Segmentation recommendation:**
- Research/benchmark stage: U-Net (VGG16 encoder) — verified numbers, strong baseline.
- On-device stage: U-Net (MobileNetV2 encoder) — reproduce and benchmark in Module 4/5.
- Skip GAN-assisted and PVT variants — marginal accuracy gain does not justify training cost or edge infeasibility.

**Encoder decision note (Module 3 → Module 4):** MobileNetV2 was chosen over ResNet34 as the on-device segmentation encoder. Reason: MobileNetV2 has ~3.5M params vs ResNet34's ~21M, is explicitly designed for edge/mobile deployment, and is TensorRT INT8 compatible. ResNet34 offers no edge-feasibility advantage over MobileNetV2 and was not benchmarked in any Jetson Nano deployment paper found in Module 2. This decision is reflected in `src/segmentation/model.py` which implements `UNetVGG16` and `UNetMobileNetV2` — ResNet34 is not implemented.

**YOLO decision point:**
YOLO is an object detector (bounding boxes), not a segmenter. Only relevant if TejaLens's camera captures a wide-field body image and needs to locate the lesion before cropping. If the device captures a tight, centered lesion crop already, skip YOLO entirely — U-Net directly on the crop is simpler and sufficient. This depends on the device's actual optics/framing — confirm with hardware spec before Module 4.

---

## 2. Classification Model Comparison

Numbers on HAM10000 and/or ISIC 2019 unless noted. "Balanced" = explicitly rebalanced split; "Imbalanced" = raw dataset distribution.

| Model | Reported Accuracy | F1 / Sensitivity | Params (approx.) | Size (MB approx.) | Jetson Nano Feasibility | Source |
|---|---|---|---|---|---|---|
| **MobileNetV2** | Lower end — weakest in most comparative studies | ~80–85% F1 (literature estimate) | ~3.5M | ~14MB | High — designed for mobile/edge, TensorRT INT8 clean | Literature estimate. Lowest accuracy ceiling of all candidates. |
| **EfficientNet-B0** | 96–97.15% (balanced split) | 95% precision, ~97% sensitivity | ~5.3M | ~20MB | High — best accuracy/size tradeoff, realistic Nano candidate | PARTIAL: MDPI NDT 2025, DOI: 10.3390/ndt3040023. Balanced HAM10000. **REPRODUCED (reports/eval_ce_baseline.json, 2026-09-19): HAM10000 val (1,502 samples, val_fraction=0.15, seed=42) — Accuracy=0.7696, Macro F1=0.7384, Mean malignant recall=0.8027 (mel=0.7943, bcc=0.9101, akiec=0.7037), nv recall=0.7418. ISIC 2019 (8-class) — Accuracy=0.7081, Macro F1=0.7342, 20 epochs.** Gap vs literature explained by imbalanced split. |
| **EfficientNet-B3/B4** | Higher than B0 on training accuracy; generalization gap widens B1→B4 | — | ~12–19M | ~48–75MB | Medium — heavier than B0, still edge-plausible with quantization | Literature estimate. Not recommended as primary on-device model. |
| **ResNet-50** | ~86–90% | — | ~25M | ~98MB | Medium — larger than EfficientNet-B0 for similar or lower accuracy | PARTIAL: DSCC_Net, PMC10093058. ResNet-152 reported 89.68% on ISIC 2020/HAM10000/DermIS combined. |
| **Plain ViT (ViT-B/16)** | ~84–90% typical | — | ~86M | ~330MB | Low — too large and data-hungry; slower without heavy optimization | Literature estimate. Not recommended for either track. |
| **Swin Transformer (small)** | ~95%+ on harder/imbalanced sets; outperforms CNNs by 10+ points on mobile-acquired images | — | ~28M | ~110MB | Low — same edge problem as ViT; worse for Maxwell GPU specifically | VERIFIED: arXiv 2509.04800, Sept 2025. Best performer on mobile-acquired images — relevant for TejaLens generalization. **REPRODUCED (reports/eval_ce_baseline.json, 2026-09-19): HAM10000 val (1,502 samples, same split as B0) — Accuracy=0.4747, Macro F1=0.5790, Mean malignant recall=0.8857 (mel=0.8800, bcc=0.9438, akiec=0.8333), BUT nv recall=0.2776 — model misclassifies 72% of nevi. NOT deployable in current state. Needs retraining. The F1=0.7393 reported during training was a mid-run checkpoint value, not a held-out eval — that number is superseded by this eval.** |
| **EfficientFormerV2** | 97.11% (balanced HAM10000) | F1 97.14%, Sensitivity 96.85%, Specificity 96.70% | Lightweight hybrid | ~30MB (literature estimate) | Medium-High — purpose-built mobile-friendly transformer hybrid | VERIFIED: Manzoor et al., DIGITAL HEALTH 2025, DOI: 10.1177/20552076251351858. Balanced split only — check imbalanced performance. |
| **GlobalSkinNet (CNN+Transformer hybrid)** | 98% HAM10000, 98% ISIC-2019, 97% ISIC-2020, 100% PH2 (caution: 200 images) | — | Not reported | Not reported | Unknown — model size not clearly reported | VERIFIED: Scientific Reports 2026, DOI: 10.1038/s41598-026-43376-0. Cross-dataset consistency is a strength. |
| **Swin-ViT + EfficientNetB4 Ensemble** | 98.5% (Eastern-population 7-class dataset) | — | Sum of member models | >200MB combined | Not edge-feasible — server-side only | VERIFIED: Bioengineering MDPI 2025, DOI: 10.3390/bioengineering12090934. Research accuracy ceiling. |
| **MaxViT-Tiny + ConvNeXt-Tiny + EfficientNetV2-B0 Ensemble** | 96% (uncertainty-filtered, HAM10000) | Precision 94%, Recall 95%, F1 95%, ROC-AUC 99% | Sum of member models | ~100MB+ combined | Not edge-feasible as-is — distill for edge | VERIFIED: arXiv 2608.11280, Aug 2026. Best uncertainty-aware result found. Grad-CAM++ + MC Dropout included. |
| **EdgeNeXt / MobileViT** | Not benchmarked on skin lesion datasets in reviewed literature | — | Low single-digit M | <10MB | High — purpose-built for constrained edge hardware including Jetson Nano | Literature estimate. Worth prototyping in Module 4/5 as a lightweight alternative to EfficientNet-B0. |
| **LMS-ViT (smartphone-focused)** | 90% (HAM10000 + smartphone images combined) | 18% improvement over CNN baselines | — | — | Unknown | VERIFIED: PMC12440931. 30% lower computational cost vs. CNN baselines. Relevant for non-dermoscopic generalization. |

---

## 3. Edge Deployment Feasibility Detail

| Model | TensorRT INT8 Compatible | Expected FPS on Jetson Nano | Accuracy Drop (INT8 vs FP32) | Notes |
|---|---|---|---|---|
| MobileNetV2 | Yes — clean | High (>30 FPS estimated) | <1% typical | VERIFIED (Orin Nano proxy): arXiv 2507.17123. Note: Orin Nano ≠ original Nano — Maxwell GPU has more limited INT8 support. |
| EfficientNet-B0 | Yes | 15–25 FPS estimated | <2% typical for CNNs | Best accuracy/edge tradeoff. Primary on-device candidate. |
| EfficientFormerV2 | Partial — hybrid transformer layers may need fallback to FP16 | 8–15 FPS estimated | Unknown — needs benchmarking | Prototype in Module 5. |
| Swin Transformer | Poor — attention layers quantize poorly on Maxwell GPU | <5 FPS estimated | Significant | Not recommended for on-device. |
| Plain ViT | Poor | <3 FPS estimated | Significant | Not recommended for on-device. |
| U-Net (MobileNetV2 encoder) | Yes | 10–20 FPS estimated | <2% typical | Recommended on-device segmentation model. |

**Important hardware note:** The two deployment papers found (arXiv 2507.17123, 2507.17125) use Jetson **Orin** Nano (Ampere GPU, 8GB RAM) — not the original Jetson Nano (Maxwell GPU, 4GB RAM) specified for TejaLens. Orin Nano has significantly better INT8/TensorRT support. The FPS numbers above are estimates adjusted downward for Maxwell. Real numbers must be measured in Module 5 — do not present the Orin Nano paper numbers as directly applicable to TejaLens.

---

## 4. Recommendation

### Research / Benchmark Model (GPU training, Colab)
**Primary: Swin Transformer (small variant) or EfficientFormerV2**
- Swin outperforms all plain CNNs on harder/imbalanced sets and on mobile-acquired images (directly relevant to TejaLens)
- EfficientFormerV2 has verified numbers on HAM10000 (97.11%) and is part of a published two-stage pipeline with U-Net/VGG16
- Report both in the literature comparison; pick the one with better reproduced numbers on your own data split

**Upper bound reference: Swin+EfficientNetB4 ensemble (98.5%) or MaxViT+ConvNeXt+EfficientNetV2 ensemble (96%, 99% AUC)**
- Use as the accuracy ceiling in the comparison table
- Do not attempt to deploy either on Jetson Nano

### On-Device / Deployment Model (Jetson Nano)
**Primary: EfficientNet-B0 (TensorRT INT8)**
- Best verified accuracy (~97%) at smallest feasible size (~5.3M params, ~20MB)
- Quantizes cleanly to INT8
- Supports MC Dropout (keep dropout active at inference) and Grad-CAM

**Fallback: MobileNetV2 (TensorRT INT8)**
- Lower accuracy ceiling but highest edge feasibility
- Use if EfficientNet-B0 fails to meet latency requirements on the Nano

**Prototype: EdgeNeXt or MobileViT**
- Purpose-built for constrained edge hardware
- No verified skin lesion numbers yet — benchmark in Module 4/5

### Segmentation Model
**Research stage: U-Net (VGG16 encoder)** — verified 97.59% / 89.12% Jaccard / 94.24% Dice
**On-device stage: U-Net (MobileNetV2 encoder)** — reproduce and benchmark in Module 4/5

---

## 5. Module 4 Reproduced Results Summary

_All numbers sourced from `reports/eval_ce_baseline.json` (timestamp: 2026-09-19T12:36:01, device: cuda, val_split: HAMValDataset val_fraction=0.15 seed=42, 1,502 samples). Swin-Small v2 numbers from `reports/eval_swin_v2.json` (timestamp: 2026-09-20T18:20:01). Segmentation Dice sourced from Kaggle training terminal output — no saved metrics file exists for it yet._

| Model | Dataset | Accuracy | Macro F1 | Mean Malignant Recall | nv Recall | Dice | Status |
|---|---|---|---|---|---|---|---|
| U-Net (VGG16) | ISIC 2018 Task 1 | — | — | — | — | **0.9043** | ✓ Deployable |
| EfficientNet-B0 | HAM10000 (7-class) | **0.7696** | **0.7384** | **0.8027** | 0.7418 | — | ✓ Deployable |
| Swin-Small (original) | HAM10000 (7-class) | 0.4747 | 0.5790 | 0.8857 | 0.2776 | — | ✗ nv collapse |
| Swin-Small v2 (retrain lr=1e-5, 23 epochs) | HAM10000 (7-class) | 0.4015 | 0.4496 | 0.8069 | 0.2378 | — | ✗ Worse — retrain failed |
| EfficientNet-B0 | ISIC 2019 (8-class) | 0.7081 | 0.7342 | — | — | — | ✓ Trained |

**Key findings:**
- EfficientNet-B0 is the only deployable classifier — balanced across all classes
- Swin-Small nv collapse persists across both training runs. v2 retrain (lr=1e-5, weighted CE, 23 epochs) made things worse: F1 dropped from 0.5790 → 0.4496, nv recall dropped from 0.2776 → 0.2378. The model overfits minority classes (df recall=1.0, vasc=0.9687) while further collapsing nv
- Swin-Small is not suitable for HAM10000 without significant regularization changes (label smoothing, stronger augmentation, or frozen encoder fine-tuning). Closed as research-only benchmark — original checkpoint retained
- Focal loss retrain of EfficientNet-B0 also failed (reports/eval_focal.json: acc=0.3129, nv recall=0.0786) — weights in experiments/
- Gap vs literature (97%+) explained by imbalanced splits — literature uses balanced/resampled data

**Open items before Module 4 is fully closed:**
1. ~~Retrain Swin-Small~~ — closed, both attempts failed, original checkpoint retained as research benchmark
2. Save U-Net segmentation metrics to a JSON file (currently only in training terminal output)
3. Test EfficientNet-B0 on ISIC 2024 SLICE-3D as held-out generalization test
