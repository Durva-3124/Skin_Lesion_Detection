# Quality Gate Test Results
_Run: 2026-09-21T00:50:17 | Device: cpu | Method: Laplacian variance + brightness thresholds_

8 images tested: 6 real HAM10000 dermoscopic images + 2 synthetic degraded images.
Synthetic blurry = Gaussian blur (σ=30) applied to ISIC_0033301.
Synthetic dark = 5% brightness of ISIC_0033350.

| Image ID | Type | Result | Detail |
|---|---|---|---|
| ISIC_0033301 | real | **PASS** | ok (sharpness=64.2, brightness=143.8) |
| ISIC_0033350 | real | **FAIL** | blurry (sharpness=30.3) |
| ISIC_0033400 | real | **FAIL** | blurry (sharpness=20.9) |
| ISIC_0033450 | real | **PASS** | ok (sharpness=57.5, brightness=145.4) |
| ISIC_0033500 | real | **FAIL** | blurry (sharpness=45.4) |
| ISIC_0033550 | real | **PASS** | ok (sharpness=475.6, brightness=162.2) |
| synthetic_blurry | synthetic_degraded | **FAIL** | blurry (sharpness=1.3) |
| synthetic_dark | synthetic_degraded | **FAIL** | blurry (sharpness=0.6) |

**Summary:** 3 PASS, 5 FAIL

**Thresholds (from `_quality_check` in `src/pipeline.py`):**
- Laplacian variance < 50 → reject as blurry
- Mean brightness < 30 → reject as too dark
- Mean brightness > 225 → reject as overexposed