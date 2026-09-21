# MC Dropout Uncertainty Test Results
_Run: 2026-09-21T00:50:17 | Device: cpu | MC passes: 30 | Model: efficientnet_b0_ham10000.pth_

Uncertainty = mean variance across 30 MC Dropout forward passes (lower = more confident).
Segmentation coverage = fraction of pixels with mask probability > 0.5 (UNet-VGG16, 512px input).

| Image ID | Predicted Class | Confidence | Uncertainty | Risk | Seg Coverage |
|---|---|---|---|---|---|
| ISIC_0033301 | bcc | 0.5386 | 0.000741 | MEDIUM | 0.656 |
| ISIC_0033350 | REJECTED | — | — | blurry (sharpness=30.3) | — |
| ISIC_0033400 | REJECTED | — | — | blurry (sharpness=20.9) | — |
| ISIC_0033450 | vasc | 1.0000 | 0.000000 | LOW | 0.322 |
| ISIC_0033500 | REJECTED | — | — | blurry (sharpness=45.4) | — |
| ISIC_0033550 | bkl | 0.5110 | 0.001728 | MEDIUM | 0.567 |
| synthetic_blurry | REJECTED | — | — | blurry (sharpness=1.3) | — |
| synthetic_dark | REJECTED | — | — | blurry (sharpness=0.6) | — |

**Notes:**
- Rejected images (quality gate failures) show no inference results.
- Uncertainty typical range: 0.0001–0.005 for confident predictions.
- HIGH risk = malignant class (mel/bcc/akiec) with confidence ≥ 0.70.
- MEDIUM risk = malignant class < 0.70 confidence, or low-confidence benign.
- LOW risk = benign class with confidence ≥ 0.70.
- Grad-CAM overlays saved to `reports/gradcam_samples/`.