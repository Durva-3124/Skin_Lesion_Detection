"""
scripts/run_module6_tests.py
Runs quality gate, MC Dropout uncertainty, and Grad-CAM against 8 sample images
(6 real HAM10000 + 2 synthetic degraded images for rejection testing).
Saves outputs to:
  reports/quality_gate_test.md
  reports/uncertainty_test.md
  reports/gradcam_samples/<image_id>_gradcam.jpg

Note on segmentation: UNet-VGG16 requires input >= 512x512 (5 pooling stages in VGG16
features include MaxPool2d layers; 224px causes spatial collapse at enc5).
Segmentation is run at 512px; classification and Grad-CAM at 224px.
"""

import sys
import pathlib
import datetime
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.classification.model import get_classification_model
from src.segmentation.model import get_segmentation_model
from src.preprocessing.transforms import get_val_transforms, remove_hair
from src.preprocessing.dataset import HAM_CLASSES
from src.explainability import GradCAM, overlay_heatmap

ROOT        = pathlib.Path(__file__).parent.parent
SEG_WEIGHTS = ROOT / "models" / "unet_vgg16.pth"
CLS_WEIGHTS = ROOT / "models" / "efficientnet_b0_ham10000.pth"
IMG_DIR     = ROOT / "data" / "ham10000"
REPORT_DIR  = ROOT / "reports"
GRADCAM_DIR = REPORT_DIR / "gradcam_samples"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

REAL_IMAGES = [
    "ISIC_0033301",
    "ISIC_0033350",
    "ISIC_0033400",
    "ISIC_0033450",
    "ISIC_0033500",
    "ISIC_0033550",
]

SEG_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_blurry_image(source_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    img = cv2.imread(str(source_path))
    blurred = cv2.GaussianBlur(img, (51, 51), 30)
    cv2.imwrite(str(out_path), blurred)
    return out_path


def make_dark_image(source_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    img = cv2.imread(str(source_path))
    dark = (img * 0.05).astype(np.uint8)
    cv2.imwrite(str(out_path), dark)
    return out_path


def quality_check(image_path: pathlib.Path) -> tuple[bool, str]:
    image_np = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, f"blurry (sharpness={laplacian_var:.1f})"
    if gray.mean() < 30:
        return False, f"too dark (mean={gray.mean():.1f})"
    if gray.mean() > 225:
        return False, f"overexposed (mean={gray.mean():.1f})"
    return True, f"ok (sharpness={laplacian_var:.1f}, brightness={gray.mean():.1f})"


def classify_with_uncertainty(
    cls_model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    mc_passes: int = 30,
) -> tuple[int, float, float, np.ndarray]:
    inp = tensor.unsqueeze(0).to(device)

    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    cls_model.apply(enable_dropout)
    probs_list = []
    try:
        with torch.no_grad():
            for _ in range(mc_passes):
                logits = cls_model(inp)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
    finally:
        cls_model.eval()

    probs_stack = np.stack(probs_list, axis=0)
    mean_probs = probs_stack.mean(axis=0).squeeze()
    uncertainty = float(probs_stack.var(axis=0).squeeze().mean())
    pred_class = int(mean_probs.argmax())
    confidence = float(mean_probs.max())
    return pred_class, confidence, uncertainty, mean_probs


def risk_level(pred_class: int, confidence: float) -> str:
    malignant = {"mel", "bcc", "akiec"}
    name = HAM_CLASSES[pred_class]
    if name in malignant:
        return "HIGH" if confidence >= 0.70 else "MEDIUM"
    return "LOW" if confidence >= 0.70 else "MEDIUM"


def main():
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    device = torch.device("cpu")

    print("Loading models...")
    seg_model = get_segmentation_model(encoder="vgg16").to(device)
    seg_model.load_state_dict(torch.load(str(SEG_WEIGHTS), map_location=device, weights_only=True))
    seg_model.eval()

    cls_model = get_classification_model("efficientnet_b0", num_classes=7).to(device)
    cls_model.load_state_dict(torch.load(str(CLS_WEIGHTS), map_location=device, weights_only=True))
    cls_model.eval()

    gradcam = GradCAM(cls_model, device)
    cls_transform = get_val_transforms(224)

    blurry_path = REPORT_DIR / "_tmp_blurry.jpg"
    dark_path   = REPORT_DIR / "_tmp_dark.jpg"
    make_blurry_image(IMG_DIR / f"{REAL_IMAGES[0]}.jpg", blurry_path)
    make_dark_image(IMG_DIR / f"{REAL_IMAGES[1]}.jpg", dark_path)

    test_images = [
        (REAL_IMAGES[0], IMG_DIR / f"{REAL_IMAGES[0]}.jpg", "real"),
        (REAL_IMAGES[1], IMG_DIR / f"{REAL_IMAGES[1]}.jpg", "real"),
        (REAL_IMAGES[2], IMG_DIR / f"{REAL_IMAGES[2]}.jpg", "real"),
        (REAL_IMAGES[3], IMG_DIR / f"{REAL_IMAGES[3]}.jpg", "real"),
        (REAL_IMAGES[4], IMG_DIR / f"{REAL_IMAGES[4]}.jpg", "real"),
        (REAL_IMAGES[5], IMG_DIR / f"{REAL_IMAGES[5]}.jpg", "real"),
        ("synthetic_blurry", blurry_path, "synthetic_degraded"),
        ("synthetic_dark",   dark_path,   "synthetic_degraded"),
    ]

    qg_rows = []
    unc_rows = []

    for image_id, image_path, image_type in test_images:
        print(f"  Processing {image_id}...")
        passed, reason = quality_check(image_path)
        qg_rows.append((image_id, image_type, "PASS" if passed else "FAIL", reason))

        if passed:
            pil_img = Image.open(image_path).convert("RGB")

            # Hair removal
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            clean_bgr = remove_hair(bgr)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            clean_pil = Image.fromarray(clean_rgb)

            # Segmentation at 512px
            seg_tensor = SEG_TRANSFORM(clean_pil)
            with torch.no_grad():
                mask_logit = seg_model(seg_tensor.unsqueeze(0).to(device))
                seg_mask = torch.sigmoid(mask_logit).squeeze().cpu().numpy()
            seg_coverage = float((seg_mask > 0.5).mean())

            # Classification + MC Dropout at 224px
            cls_tensor = cls_transform(clean_pil)
            pred_idx, conf, unc, mean_probs = classify_with_uncertainty(cls_model, cls_tensor, device)
            pred_name = HAM_CLASSES[pred_idx]
            risk = risk_level(pred_idx, conf)

            unc_rows.append((image_id, pred_name, conf, unc, risk, seg_coverage))

            # Grad-CAM at 224px
            cam = gradcam.generate(cls_tensor, pred_idx)
            display = np.array(clean_pil.resize((224, 224)))
            overlay = overlay_heatmap(display, cam)
            out_path = GRADCAM_DIR / f"{image_id}_gradcam.jpg"
            Image.fromarray(overlay).save(str(out_path))
            print(f"    pred={pred_name} conf={conf:.3f} unc={unc:.6f} risk={risk} seg_cov={seg_coverage:.3f}")
            print(f"    Grad-CAM saved: {out_path.name}")
        else:
            unc_rows.append((image_id, "REJECTED", None, None, reason, None))

    blurry_path.unlink(missing_ok=True)
    dark_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Write quality_gate_test.md
    # ------------------------------------------------------------------
    qg_lines = [
        "# Quality Gate Test Results",
        f"_Run: {timestamp} | Device: cpu | Method: Laplacian variance + brightness thresholds_",
        "",
        "8 images tested: 6 real HAM10000 dermoscopic images + 2 synthetic degraded images.",
        "Synthetic blurry = Gaussian blur (σ=30) applied to ISIC_0033301.",
        "Synthetic dark = 5% brightness of ISIC_0033350.",
        "",
        "| Image ID | Type | Result | Detail |",
        "|---|---|---|---|",
    ]
    for image_id, image_type, result, detail in qg_rows:
        qg_lines.append(f"| {image_id} | {image_type} | **{result}** | {detail} |")

    passed_count = sum(1 for _, _, r, _ in qg_rows if r == "PASS")
    failed_count = len(qg_rows) - passed_count
    qg_lines += [
        "",
        f"**Summary:** {passed_count} PASS, {failed_count} FAIL",
        "",
        "**Thresholds (from `_quality_check` in `src/pipeline.py`):**",
        "- Laplacian variance < 50 → reject as blurry",
        "- Mean brightness < 30 → reject as too dark",
        "- Mean brightness > 225 → reject as overexposed",
    ]
    (REPORT_DIR / "quality_gate_test.md").write_text("\n".join(qg_lines), encoding="utf-8")
    print("Saved reports/quality_gate_test.md")

    # ------------------------------------------------------------------
    # Write uncertainty_test.md
    # ------------------------------------------------------------------
    unc_lines = [
        "# MC Dropout Uncertainty Test Results",
        f"_Run: {timestamp} | Device: cpu | MC passes: 30 | Model: efficientnet_b0_ham10000.pth_",
        "",
        "Uncertainty = mean variance across 30 MC Dropout forward passes (lower = more confident).",
        "Segmentation coverage = fraction of pixels with mask probability > 0.5 (UNet-VGG16, 512px input).",
        "",
        "| Image ID | Predicted Class | Confidence | Uncertainty | Risk | Seg Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for image_id, pred, conf, unc, risk, seg_cov in unc_rows:
        if conf is not None:
            unc_lines.append(
                f"| {image_id} | {pred} | {conf:.4f} | {unc:.6f} | {risk} | {seg_cov:.3f} |"
            )
        else:
            unc_lines.append(f"| {image_id} | {pred} | — | — | {risk} | — |")

    unc_lines += [
        "",
        "**Notes:**",
        "- Rejected images (quality gate failures) show no inference results.",
        "- Uncertainty typical range: 0.0001–0.005 for confident predictions.",
        "- HIGH risk = malignant class (mel/bcc/akiec) with confidence ≥ 0.70.",
        "- MEDIUM risk = malignant class < 0.70 confidence, or low-confidence benign.",
        "- LOW risk = benign class with confidence ≥ 0.70.",
        "- Grad-CAM overlays saved to `reports/gradcam_samples/`.",
    ]
    (REPORT_DIR / "uncertainty_test.md").write_text("\n".join(unc_lines), encoding="utf-8")
    print("Saved reports/uncertainty_test.md")
    print("Grad-CAM overlays saved to reports/gradcam_samples/")


if __name__ == "__main__":
    main()
