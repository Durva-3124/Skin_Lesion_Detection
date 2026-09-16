# Module 2 — Literature Review
# TejaLens Skin Cancer Pre-Screening Module

_Organized under four headings matching the project brief.
Each entry: Objective · Dataset · Model · Results · Strengths · Weaknesses._

---

## 1. Skin Cancer Detection

### EfficientNet-B0 for Dermoscopic Lesion Classification
- **Objective:** Multi-class skin lesion classification optimized for accuracy/efficiency tradeoff.
- **Dataset:** ISIC dermoscopic dataset (balanced split).
- **Model:** EfficientNet-B0 (fine-tuned, ImageNet pretrained).
- **Results:** ~97% accuracy, 95% precision, 99% sensitivity.
- **Strengths:** Lightweight enough to consider for edge deployment; high sensitivity is critical for a pre-screening tool. Compound scaling makes it more parameter-efficient than ResNet/VGG families.
- **Weaknesses:** Sensitive to hyperparameter tuning and preprocessing quality (hair removal, color normalization). Performance degrades noticeably on non-dermoscopic (smartphone) images — a direct concern for TejaLens.
- **Source:** Multiple comparative studies on ISIC benchmarks (arXiv/PubMed — verify exact DOI before citing in final report).

### Swin-ViT + EfficientNetB4 Ensemble
- **Objective:** Maximum-accuracy multi-class classification via ensemble voting.
- **Dataset:** Eastern-population 7-class dermoscopic dataset.
- **Model:** Swin Transformer + EfficientNetB4 hard/soft voting ensemble.
- **Results:** ~98.5% test accuracy — highest reported ensemble result in surveyed literature.
- **Strengths:** Best raw accuracy found; soft voting ensemble reduces individual model errors.
- **Weaknesses:** Computationally expensive — two large models running in parallel. Completely infeasible on Jetson Nano (128 CUDA cores, 4GB RAM). This is the research/benchmark ceiling, not the deployment model. Flag this tension explicitly in Module 3.
- **Source:** arXiv (verify DOI — search "Swin EfficientNet skin lesion ensemble").

### CNN vs. Transformer Comparison on Mobile-Acquired Images
- **Objective:** Benchmark plain CNNs vs. Swin Transformer vs. plain ViT on harder, more realistic datasets including mobile-acquired images.
- **Dataset:** Mixed — ISIC benchmarks + mobile-acquired skin lesion images.
- **Model:** ResNet50, VGG19, plain ViT, Swin Transformer.
- **Results:** Plain CNNs (ResNet50, VGG19) plateau at 84–90% on harder/imbalanced sets; plain ViT similar range. Swin Transformer and CNN-Transformer hybrids outperform both, often by 10+ points on harder sets. On mobile-acquired images specifically, CNNs plateaued near 60% while Swin/ViT reached higher validation accuracy but showed more overfitting.
- **Strengths:** Directly relevant to TejaLens — mobile-acquired image results are the closest proxy to what TejaLens's camera will capture.
- **Weaknesses:** Swin overfits more on small mobile datasets; needs strong augmentation. Plain ViT requires large datasets to generalize.
- **Source:** Multiple 2023–2025 comparative studies (arXiv — search "skin lesion classification mobile acquired Swin Transformer").

---

## 2. Lesion Segmentation

### U-Net with VGG16 Encoder (Standard Baseline)
- **Objective:** Binary lesion/background segmentation as a preprocessing stage before classification.
- **Dataset:** ISIC 2018 Task 1 (2,594 images + pixel-level masks).
- **Model:** U-Net with VGG16 encoder (ImageNet pretrained).
- **Results:** ~97.6% segmentation accuracy, ~89% Jaccard (IoU), ~94% Dice coefficient.
- **Strengths:** Well-established baseline; VGG16 encoder is pretrained and stable. Dice/Jaccard scores are strong enough for a pre-screening segmentation stage. Paired with EfficientFormerV2 classifier, the full pipeline achieved ~97.1% classification accuracy on HAM10000.
- **Weaknesses:** VGG16 encoder is relatively large — may need to swap to a lighter encoder (MobileNetV2, EfficientNet-B0) for Jetson Nano deployment. Segmentation quality degrades on low-contrast or hair-occluded lesions.
- **Source:** EfficientFormerV2 + U-Net(VGG16) pipeline paper (arXiv/PubMed — verify DOI).

### GAN-Assisted and PVT-Augmented U-Net Variants
- **Objective:** Push segmentation accuracy beyond the standard U-Net baseline using generative augmentation or attention mechanisms.
- **Dataset:** ISIC 2016, 2017, 2018.
- **Model:** U-Net variants with GAN-based data augmentation or Pyramid Vision Transformer (PVT) encoder.
- **Results:** Mid-to-high 90s% Dice/Jaccard across ISIC datasets — marginal improvement over standard U-Net baseline.
- **Strengths:** GAN augmentation helps with class imbalance and rare lesion types. PVT encoder captures global context better than VGG16.
- **Weaknesses:** Higher training cost and complexity for marginal gain over standard U-Net. GAN training is unstable. PVT variants are heavier than VGG16 — worse for edge deployment.
- **Source:** Multiple 2022–2024 segmentation papers (arXiv — search "skin lesion segmentation PVT U-Net ISIC").

---

## 3. Skin Classification

### EfficientFormerV2 on HAM10000
- **Objective:** Efficient transformer-based classification on the standard 7-class benchmark.
- **Dataset:** HAM10000 (balanced split).
- **Model:** EfficientFormerV2 (hybrid CNN-Transformer).
- **Results:** ~97.1% accuracy, ~97.1% F1, ~96.9% sensitivity, ~96.7% specificity.
- **Strengths:** Hybrid architecture captures both local texture (CNN) and global context (Transformer). Strong F1 on a balanced split. Paired with U-Net(VGG16) segmentation front-end in the same pipeline.
- **Weaknesses:** Results are on a balanced split — real HAM10000 is heavily imbalanced (NV ~67%). Performance on the raw imbalanced set will be lower. Verify sensitivity on minority classes (DF, VASC) specifically.
- **Source:** EfficientFormerV2 skin classification paper (arXiv — verify DOI).

### GlobalSkinNet (Hybrid Global Contextual Vision Transformer)
- **Objective:** High-accuracy multi-dataset skin lesion classification using a CNN + Transformer hybrid.
- **Dataset:** PH2, ISIC 2019, ISIC 2020, HAM10000.
- **Model:** Custom hybrid combining CNN feature extraction with global Transformer attention blocks ("GlobalSkinNet").
- **Results:** Near-ceiling on PH2 (caution: only 200 images), ~97–98% on ISIC 2019 and HAM10000 — consistent with other hybrid-model literature.
- **Strengths:** Tested across multiple datasets — cross-dataset consistency is more convincing than single-dataset results. Global attention helps with subtle lesion features.
- **Weaknesses:** PH2 result is unreliable due to tiny dataset size (200 images). Model size not reported clearly — edge feasibility unknown. Needs independent replication.
- **Source:** GlobalSkinNet paper (arXiv 2024 — search "GlobalSkinNet skin lesion classification hybrid transformer").

### Plain CNN Baselines (ResNet50, VGG19)
- **Objective:** Establish lower-bound baselines for comparison.
- **Dataset:** ISIC benchmarks, HAM10000.
- **Model:** ResNet50, VGG19 (fine-tuned).
- **Results:** 84–90% accuracy on harder/imbalanced sets.
- **Strengths:** Well-understood, fast to train, easy to deploy on edge hardware. ResNet50 is feasible on Jetson Nano with TensorRT quantization.
- **Weaknesses:** Accuracy ceiling is too low for a pre-screening tool where missed melanomas are the critical failure mode. Not recommended as the primary model.
- **Source:** Standard benchmark literature (widely replicated).

---

## 4. AI in Dermatology

### Monte Carlo Dropout for Uncertainty Estimation
- **Objective:** Produce per-prediction uncertainty scores alongside classification output.
- **Dataset:** Pigmented lesion datasets (ISIC variants).
- **Model:** Any CNN/Transformer with dropout layers kept active at inference (MC Dropout).
- **Results:** Provides usable uncertainty estimates; multiple stochastic forward passes (typically 30–50), mean = prediction, variance = uncertainty score.
- **Strengths:** Simple to implement — no architecture change needed, just keep dropout active at inference. Adds uncertainty quantification with minimal overhead.
- **Weaknesses:** Tends to be poorly calibrated on its own — overconfident on out-of-distribution inputs. Requires multiple forward passes (latency cost on Jetson Nano). Needs temperature scaling or isotonic regression for calibration.
- **Source:** Multiple 2024–2026 uncertainty papers on pigmented lesion datasets (arXiv — search "Monte Carlo Dropout skin lesion uncertainty").

### Conformal Prediction for Calibrated Uncertainty
- **Objective:** Produce statistically guaranteed, well-calibrated prediction sets with coverage guarantees.
- **Dataset:** Pigmented lesion datasets.
- **Model:** Any classifier wrapped with conformal prediction post-processing.
- **Results:** Repeatedly outperforms MC Dropout and Evidential Deep Learning for calibrated, distribution-free uncertainty on pigmented-lesion datasets.
- **Strengths:** Distribution-free coverage guarantee — the stated confidence level is actually correct, unlike raw softmax or MC Dropout. No retraining needed, applied post-hoc. Directly relevant to Module 6's risk-scoring requirement.
- **Weaknesses:** Requires a held-out calibration set. Prediction sets (not single labels) may be harder to communicate to end users. Less studied on edge-deployed models.
- **Source:** 2024–2025 conformal prediction papers on medical imaging (arXiv — search "conformal prediction skin lesion calibration").

### Ensemble + MC Dropout + Grad-CAM++ for Trust and Explainability
- **Objective:** Combine uncertainty quantification with visual explanation for clinical trustworthiness.
- **Dataset:** ISIC variants.
- **Model:** MaxViT-Tiny + ConvNeXt-Tiny + EfficientNetV2-B0 ensemble with MC Dropout and Grad-CAM++ visualization.
- **Results:** Provides both a calibrated uncertainty score and a saliency map per prediction — the combination is what makes the output clinically actionable rather than a black box.
- **Strengths:** Directly matches TejaLens Module 6 requirements. Grad-CAM++ highlights which lesion region drove the prediction — supports the "pre-screening not diagnosis" framing by showing the clinician what to look at.
- **Weaknesses:** Full ensemble is too large for Jetson Nano. For on-device deployment, scale down to a single lightweight model (EfficientNet-B0 or MobileNetV2) with MC Dropout + Grad-CAM only. See Module 5/6.
- **Source:** Recent XAI dermatology papers 2024–2025 (arXiv — search "Grad-CAM uncertainty ensemble skin lesion explainability").

### Non-Dermoscopic / Smartphone Skin Lesion Classification
- **Objective:** Classify skin lesions from consumer-camera or smartphone images rather than clinical dermatoscope images.
- **Dataset:** ISIC 2024 SLICE-3D (401,059 cropped images from 3D Total Body Photography — closest proxy to non-dermoscopic capture).
- **Model:** Various (literature is thinner than dermoscope benchmark literature).
- **Results:** Significant accuracy drop vs. dermoscope-trained models when tested on smartphone images — exact numbers vary by study. Domain gap is a known open problem.
- **Strengths:** Directly relevant to TejaLens — the device is not a dermatoscope. ISIC 2024 is the best available proxy dataset for this condition.
- **Weaknesses:** Literature is sparse compared to dermoscope benchmarks. Most published models are trained and tested on dermoscope images only — generalization to phone-camera images is rarely reported. This is a gap in the literature that TejaLens's ISIC 2024 generalization test will directly address.
- **Source:** Search specifically: "smartphone skin lesion classification non-dermoscopic" on arXiv/PubMed. ISIC 2024 challenge papers (2024–2025).

### Jetson Nano / Edge Deployment of Skin Lesion Models
- **Objective:** Deploy trained skin lesion classification models on low-power edge hardware.
- **Dataset:** N/A (deployment studies, not training studies).
- **Model:** Quantized/pruned CNN variants (MobileNetV2, EfficientNet-B0) exported via TensorRT.
- **Results:** Literature is sparse. General findings: INT8 quantization via TensorRT reduces model size ~4x and improves inference speed ~2–3x on Jetson Nano with <2% accuracy drop for CNN models. ViT/Transformer models quantize less cleanly than CNNs on Maxwell GPU.
- **Strengths:** TensorRT INT8 is the standard path for Jetson Nano deployment. MobileNetV2 and EfficientNet-B0 are the most commonly reported feasible architectures.
- **Weaknesses:** Most skin lesion papers report GPU-server accuracy only — edge feasibility is rarely benchmarked. Jetson Nano's Maxwell GPU has limited INT8 support compared to newer Jetson boards. See Module 5 for detailed deployment analysis.
- **Source:** Search: "TensorRT skin lesion Jetson Nano edge deployment" on arXiv/IEEE. NVIDIA Jetson benchmark documentation.

---

## Synthesis — Most Promising Architectures for TejaLens

Given the constraints (Jetson Nano edge deployment, multi-class classification, pre-screening not diagnosis, high sensitivity required):

**1. EfficientNet-B0 — recommended on-device model**
Consistently strong accuracy (~97%) with the smallest footprint of the high-performing models. Quantizes cleanly to INT8 via TensorRT. Feasible on Jetson Nano at usable frame rates. Pair with MC Dropout for uncertainty and Grad-CAM for explainability.

**2. EfficientFormerV2 or Swin-Transformer (small variant) — recommended research/benchmark model**
Hybrid CNN-Transformer architectures consistently outperform plain CNNs by 5–10+ points on harder/imbalanced sets and on mobile-acquired images. Use this as the GPU-trained benchmark model reported in the literature comparison. Do not deploy this on the Jetson Nano — report the accuracy gap explicitly.

**3. U-Net with VGG16 encoder — recommended segmentation stage**
Well-established, strong Dice/Jaccard on ISIC 2018, pairs cleanly with EfficientNet-B0 classifier. For on-device deployment, swap VGG16 encoder for MobileNetV2 to reduce size.

These three feed directly into Module 3's architecture comparison and Module 4's pipeline implementation decision.

---

_Next steps: Pull actual papers behind each entry from arXiv/PubMed and read methods sections before citing in final report. Search specifically for non-dermoscopic classification and Jetson Nano deployment literature — both are underrepresented above and directly relevant to TejaLens._
