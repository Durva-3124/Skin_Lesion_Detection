# Module 2 — Literature Review
## TejaLens Skin Cancer Pre-Screening Module

Updated with verified citations pulled from arXiv, PubMed, Nature/Scientific Reports, MDPI, and ScienceDirect. Each entry below is marked with a verification status.

---

### How to Read the Status Markers

- **VERIFIED**: Real paper found, title/authors/venue/DOI confirmed, numbers matched or closely matched the entry's claims.
- **PARTIAL**: A closely related real paper was found, but exact numbers in the original entry could not be independently confirmed against it — treat as a starting point, not a citation to submit as-is.
- **GAP**: No single paper matched this entry precisely enough to cite with confidence. Candidates are listed, but this needs your own follow-up read before it goes in the final report.

---

## 1. Skin Cancer Detection

### EfficientNet-B0 for Dermoscopic Lesion Classification
**Status:** `PARTIAL`

- **Full Title:** *Skin Cancer Detection and Classification Through Medical Image Analysis Using EfficientNet*
- **Venue / Year:** Nanomanufacturing and Nanotechnology / NDT (MDPI), 2025
- **DOI:** [10.3390/ndt3040023](https://doi.org/10.3390/ndt3040023)
- **Dataset:** HAM10000, rebalanced to 7,512 images across 7 classes via augmentation of minority classes and downsampling of majority classes (explicitly balanced, not the raw imbalanced HAM10000).
- **Exact Metrics Reported:** Baseline EfficientNet-B0: 77.39% accuracy $\rightarrow$ 89.36% with transfer learning (frozen base) $\rightarrow$ 96% with full fine-tuning + test-time augmentation (TTA) $\rightarrow$ 97.15% final, with TTA + Monte Carlo Dropout combined.
- **Balanced or Imbalanced:** Balanced (explicitly rebalanced dataset, not raw HAM10000 distribution).
- **Note:** This paper's 97.15% accuracy matches the entry's "~97% accuracy" claim well and even reuses MC Dropout, matching Module 6's needs. However, its own abstract does not separately report 95% precision / 99% sensitivity as stated in the original entry — those two specific numbers could not be confirmed against this source and should be re-checked against the paper directly, or the entry's numbers should be revised to what's actually reported (accuracy figures above).

---

### Swin-ViT + EfficientNetB4 Ensemble
**Status:** `VERIFIED`

- **Full Title:** *Deep Ensemble Learning for Multiclass Skin Lesion Classification*
- **Venue / Year:** Bioengineering (MDPI), vol. 12, no. 9, article 934, 2025
- **DOI:** [10.3390/bioengineering12090934](https://doi.org/10.3390/bioengineering12090934)
- **PMC ID:** PMC12467972
- **Dataset:** CSMUH dataset — Eastern-population, 7-class dermoscopic dataset; 25 pretrained CNN/ViT models fine-tuned, top 3 combined into the ensemble; validated via holdout with 5 randomized experiments.
- **Exact Metrics Reported:** Swin-ViT-EfficientNetB4 ensemble (hard/soft voting): 98.5% test accuracy — confirmed as the highest reported result in the paper's own comparison table against prior work on the same dataset.
- **Balanced or Imbalanced:** Not explicitly stated as balanced/imbalanced in the retrieved excerpt — verify this specifically when you pull the full paper, since it affects how much to trust the 98.5% figure.

---

### CNN vs. Transformer Comparison on Mobile-Acquired Images
**Status:** `VERIFIED`

- **Full Title:** *Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images*
- **Venue / Year:** arXiv, September 2025
- **arXiv ID:** [2509.04800](https://arxiv.org/abs/2509.04800)
- **Dataset:** Custom-curated dataset of 50+ skin disease categories captured with mobile devices (not dermoscope images) — directly the non-dermoscopic, mobile-acquired condition the entry describes.
- **Exact Metrics Reported:** Multiple CNNs and Transformer architectures evaluated; Swin Transformer achieves the best performance among all tested models by effectively capturing global contextual features. Grad-CAM used for interpretability. *(Exact per-model accuracy table was not visible in the retrieved excerpt — pull the full PDF for the specific accuracy numbers per architecture before citing precise figures.)*
- **Balanced or Imbalanced:** Real-world mobile-acquired dataset across 50+ categories — almost certainly imbalanced by nature; confirm exact class distribution in the full paper.
- **Note:** This single paper can serve as the citation for both this entry and the Section 4 "Non-Dermoscopic / Smartphone" entry below, since it directly addresses both.

---

## 2. Lesion Segmentation

### U-Net with VGG16 Encoder (Standard Baseline)
**Status:** `VERIFIED`

- **Full Title:** *Dual-stage segmentation and classification framework for skin lesion analysis using deep neural network*
- **Authors:** Khadija Manzoor, Nauman U. Gilal, Marco Agus, Jens Schneider
- **Venue / Year:** DIGITAL HEALTH (SAGE), 2025
- **DOI:** [10.1177/20552076251351858](https://doi.org/10.1177/20552076251351858)
- **PubMed ID:** 40666627
- **Dataset:** Segmentation stage trained/evaluated on ISIC 2018; the same pipeline also uses HAM10000 (classification) and the ISIC 2024 SLICE-3D dataset (tabular + image fusion) — this one paper covers three of your flagged gap areas at once.
- **Exact Metrics Reported:** U-Net (VGG16 encoder) on ISIC 2018: 97.59% accuracy, 89.12% Jaccard index, 94.24% Dice similarity coefficient — matches the entry's ~97.6% / ~89% / ~94% claims almost exactly.
- **Balanced or Imbalanced:** Segmentation task — not a class-balance question in the same sense; the paired classification stage (see EfficientFormerV2 entry below) explicitly tests both balanced and imbalanced splits.
- **Note:** This is the same paper as the EfficientFormerV2 entry below — the two are stages of one published pipeline, not two separate papers. Cite once, reference for both stages.

---

### GAN-Assisted and PVT-Augmented U-Net Variants
**Status:** `GAP`

- **Status:** No single paper was found matching the original entry's specific description (GAN-based augmentation OR Pyramid Vision Transformer encoder, evaluated on ISIC 2016/2017/2018 with Dice/Jaccard in the mid-to-high 90s).
- **Closest Real Candidates Found Instead:**
  1. *Skin lesion segmentation with a multiscale input fusion U-Net incorporating Res2-SE and pyramid dilated convolution* (MRP-UNet), Scientific Reports, 2025, DOI: [10.1038/s41598-025-92447-1](https://doi.org/10.1038/s41598-025-92447-1) — attention/pyramid-convolution U-Net variant, NOT GAN-based, tested on ISIC 2016/2017/2018/PH2/HAM10000, ~96% accuracy range.
  2. *SkinAttn-Net: a multi-level attention-based network for skin lesion segmentation*, Scientific Reports, 2025, DOI: [10.1038/s41598-025-33767-0](https://doi.org/10.1038/s41598-025-33767-0) — CBAM/SE/ViT-bottleneck attention U-Net, also not GAN or PVT-specific.
- **Recommendation:** Run a dedicated search specifically for *"GAN augmentation skin lesion segmentation U-Net"* and *"Pyramid Vision Transformer skin lesion segmentation"* as two separate queries — this entry may be describing two distinct papers that got merged into one literature-review entry originally. Do not cite either candidate above as a direct match without reading the full text first.

---

## 3. Skin Classification

### EfficientFormerV2 on HAM10000
**Status:** `VERIFIED`

- **Full Title:** *Dual-stage segmentation and classification framework for skin lesion analysis using deep neural network*
- **Authors:** Khadija Manzoor, Nauman U. Gilal, Marco Agus, Jens Schneider
- **Venue / Year:** DIGITAL HEALTH (SAGE), 2025 — same paper as the U-Net/VGG16 entry above
- **DOI / PubMed ID:** [10.1177/20552076251351858](https://doi.org/10.1177/20552076251351858) / PubMed 40666627
- **Dataset:** Balanced HAM10000 split (explicitly labeled "balanced" in the paper's own results); classification also evaluated on ISIC 2018 and ISIC 2024 SLICE-3D.
- **Exact Metrics Reported:** EfficientFormerV2 on balanced HAM10000: 97.11% accuracy, 97.14% F1-score, 96.85% sensitivity, 96.70% specificity — matches the entry's numbers almost to the decimal.
- **Balanced or Imbalanced:** Balanced — the paper explicitly separates balanced vs. imbalanced results, which directly supports the entry's own weakness note about needing to check imbalanced performance separately.

---

### GlobalSkinNet (Hybrid Global Contextual Vision Transformer)
**Status:** `VERIFIED`

- **Full Title:** *Advanced hybrid transformer CNN framework for improved skin lesion classification and segmentation*
- **Venue / Year:** Scientific Reports (Nature), 2026
- **DOI:** [10.1038/s41598-026-43376-0](https://doi.org/10.1038/s41598-026-43376-0)
- **PubMed ID / PMC ID:** 41840007 / PMC13121705
- **Dataset:** PH2, ISIC-2019, ISIC-2020, HAM10000 (four datasets, as the entry states).
- **Exact Metrics Reported:** GlobalSkinNet accuracy: 100% on PH2, 98% on ISIC-2019, 97% on ISIC-2020, 98% on HAM10000 — closely matches the entry's "near-ceiling PH2 (caution: small dataset), ~97–98% on ISIC 2019 / HAM10000" summary.
- **Balanced or Imbalanced:** ISIC-2020 is explicitly noted in the paper as typically highly imbalanced, and the paper reports GlobalSkinNet maintaining strong performance on it specifically — worth citing as a strength.
- **Note:** The paper also proposes a companion segmentation model (SkinFormNet, SegFormer + U-Net) evaluated on 5 datasets — could be cross-referenced with your segmentation section if useful.

---

### Plain CNN Baselines (ResNet50, VGG19)
**Status:** `PARTIAL`

- **Status:** This entry is a general, widely-replicated finding rather than a single paper's claim, so no single DOI fully "owns" the 84–90% figure — but a solid representative source was found.
- **Representative Source:** *DSCC_Net: Multi-Classification Deep Learning Models for Diagnosing of Skin Cancer Using Dermoscopic Images*, PMC10093058
- **Exact Metrics Reported:** Baseline comparison table: ResNet-152 89.68%, VGG-19 91.68–92.51% (two reported values across the paper's tables), MobileNet 91.46–92.51%, EfficientNet-B0 89.46%, Inception-V3 91.82%, on ISIC 2020 / HAM10000 / DermIS combined, with SMOTE-Tomek used to address minority-class imbalance.
- **Balanced or Imbalanced:** Imbalanced originally; SMOTE-Tomek applied as a correction — worth noting since the entry's 84–90% ceiling claim is dataset/preprocessing-dependent, not a fixed architectural ceiling.
- **Recommendation:** Since this is presented as a lower-bound baseline rather than a specific model contribution, it's reasonable to cite 2–3 sources showing the same pattern rather than one "authoritative" paper — the DSCC_Net numbers above are a solid anchor.

---

## 4. AI in Dermatology

### Monte Carlo Dropout for Uncertainty Estimation
**Status:** `VERIFIED`

- **Full Title:** *Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification*
- **Authors:** Jamil Fayyad et al.
- **Venue / Year:** Computer Methods and Programs in Biomedicine (ScienceDirect) / arXiv, 2023–2024
- **DOI / arXiv ID:** ScienceDirect: S0169-2607(24)00226-8 · arXiv: [2312.07460](https://arxiv.org/abs/2312.07460)
- **Dataset:** Pigmented lesion datasets (ISIC variants); ResNet-18 as the base architecture for all three compared uncertainty methods.
- **Exact Metrics Reported:** Implements and empirically compares MC Dropout, Conformal Prediction, and Evidential Deep Learning on the same base model and datasets — directly matches the entry's description of MC Dropout methodology (multiple stochastic forward passes, mean = prediction, variance = uncertainty).
- **Balanced or Imbalanced:** Not specified in the retrieved excerpt — check the full paper's dataset section.
- **Note:** This is the **SAME** paper as the Conformal Prediction entry directly below — it compares both methods head-to-head, so it can anchor both entries with one citation.

---

### Conformal Prediction for Calibrated Uncertainty
**Status:** `VERIFIED`

- **Full Title:** *Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification*
- **Authors / Venue / DOI:** Jamil Fayyad et al., Computer Methods and Programs in Biomedicine / arXiv: [2312.07460](https://arxiv.org/abs/2312.07460) *(same paper as above)*
- **Exact Finding:** The paper's own conclusion states Conformal Prediction is a more robust, distribution-free uncertainty quantification approach compared to MC Dropout and Evidential Deep Learning in this medical-imaging setting — directly matches the entry's claim that CP "repeatedly outperforms MC Dropout and Evidential Deep Learning".
- **Additional Supporting Papers Found:**
  - *Domain Adaptive Skin Lesion Classification via Conformal Ensemble of Vision Transformers* (arXiv: [2505.15997](https://arxiv.org/abs/2505.15997), 2025) — reports 90.38% coverage on a mixed-source test set, a 9.95-point improvement over a single-source model.
  - *Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics* (arXiv: [2503.23819](https://arxiv.org/abs/2503.23819), 2025) — applies conformal prediction to a ViT foundation model (Google DermFoundation) for fairness auditing across sex/age/ethnicity.
- **Balanced or Imbalanced:** The CE-ViTs paper (2505.15997) explicitly addresses class imbalance via F1-based sampling — worth citing if imbalance-robustness is a point you want to make.

---

### Ensemble + MC Dropout + Grad-CAM++ for Trust and Explainability
**Status:** `VERIFIED` *(exact architecture match)*

- **Full Title:** *Uncertainty-Aware and Explainable Ensemble Deep Learning Framework for Multi-Class Skin Lesion Classification*
- **Venue / Year:** arXiv, August 2026
- **arXiv ID:** [2608.11280](https://arxiv.org/abs/2608.11280)
- **Dataset:** HAM10000
- **Model:** Deep ensemble of MaxViT-Tiny + ConvNeXt-Tiny + EfficientNetV2-B0 — this is an exact match to the entry's model description, word for word.
- **Exact Metrics Reported:** 96% accuracy and 99% ROC-AUC under uncertainty-aware filtering (entropy $< 1.0$, confidence $\ge 0.7$); macro-average precision 94%, recall 95%, F1-score 95%; 96% weighted-average across metrics.
- **Balanced or Imbalanced:** Not confirmed in the retrieved excerpt — check the full paper.
- **Note:** This paper is an exceptionally close match — same three architectures, same MC Dropout + Grad-CAM++ combination, same dataset. High-confidence citation.

---

### Non-Dermoscopic / Smartphone Skin Lesion Classification
**Status:** `VERIFIED`

- **Primary Source:** Same as Section 1: *Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images*, arXiv: [2509.04800](https://arxiv.org/abs/2509.04800), Sept 2025 — 50+ skin disease categories, mobile-device images, Swin Transformer best performer, Grad-CAM for interpretability.
- **Secondary Source:** *LMS-ViT: a multi-scale vision transformer approach for real-time smartphone-based skin cancer detection*, PMC12440931 — combines HAM10000 with smartphone-captured images; LMS-ViT achieves 90% accuracy, an 18% improvement over CNN baselines, with 30% lower computational cost, aimed at real-time smartphone deployment.
- **Balanced or Imbalanced:** Both papers use real-world mobile-acquired data; check each paper's methods section for exact class balance, but neither is a curated balanced benchmark like HAM10000.
- **Note:** You now have 2 papers for this gap, meeting the "at least 2–3 papers on non-dermoscopic images" target. A third search specifically for the Dascalu et al. 2022 paper (*J Cancer Res Clin Oncol* 148:2497–2505, DOI: [10.1007/s00432-021-03809-x](https://doi.org/10.1007/s00432-021-03809-x), comparing dermoscopic vs. smartphone images) is worth doing — it was seen only as a citation in another paper's reference list here, not independently confirmed, so don't cite it until you've pulled it directly.

---

### Jetson Nano / Edge Deployment of Skin Lesion Models
**Status:** `PARTIAL` *(important hardware distinction)*

- **Important Flag:** Both real papers found below deploy on the **Jetson ORIN Nano** (Ampere GPU, 8GB RAM), NOT the original **Jetson Nano** (Maxwell GPU, 4GB RAM, 128 CUDA cores) that TejaLens's constraints section specifies. These are different boards with meaningfully different INT8/TensorRT support — cite these for methodology, but flag the hardware difference explicitly in Module 5 rather than presenting the results as directly transferable.
- **Source 1:** *Computer Vision for Real-Time Monkeypox Diagnosis on Embedded Systems*, arXiv: [2507.17123](https://arxiv.org/abs/2507.17123), 2025 — MobileNetV2 on Jetson Orin Nano, TensorRT FP32/FP16/INT8 quantization, Monkeypox Skin Lesion Dataset (MSLD), 93.07% F1, up to ~2.52x inference speedup with power savings, minimal accuracy loss.
- **Source 2:** *Model Compression Engine for Wearable Devices Skin Cancer Diagnosis*, arXiv: [2507.17125](https://arxiv.org/abs/2507.17125), 2025 — same NSF-grant research group, similar TensorRT compression analysis on skin lesion data, 87.18% F1 after optimization; notes that INT8 precision underperformed FP32 in their specific setup due to the device's RAM-constrained calibration, a useful caveat for your own deployment analysis.
- **Balanced or Imbalanced:** Not specified in the retrieved excerpts — these are deployment/systems papers rather than classification-accuracy papers, so class balance is a secondary concern for them.
- **Recommendation:** For a citation that actually matches your exact hardware, search specifically for *"Jetson Nano Maxwell TensorRT INT8 CNN"* or check NVIDIA's own Jetson Nano benchmark documentation directly — the literature on the older/smaller Jetson Nano specifically (vs. Orin Nano) is thin, which the original entry's "literature is sparse" note already anticipated correctly.

---

## Summary — What Still Needs Your Attention

1. **GAN-Assisted / PVT-Augmented U-Net (Section 2):** No verified match — needs a fresh, narrower search, possibly as two separate entries.
2. **EfficientNet-B0 Entry's Specific Metrics:** 95% precision / 99% sensitivity figures were not confirmed against the matched paper — verify directly or revise to the paper's actual reported numbers.
3. **Jetson Nano Deployment:** Only Jetson ORIN Nano papers were found — flag this hardware distinction explicitly wherever you cite them.
4. **Swin-ViT + EfficientNetB4 Ensemble and Ensemble+MCDropout+GradCAM++ Entries:** Balanced/imbalanced status not confirmed from the excerpt — a quick check of the full paper will close this.
5. **Dascalu et al. 2022 (Dermoscopic vs. Smartphone Comparison):** Seen only as a citation elsewhere, not independently confirmed — pull it directly before citing.