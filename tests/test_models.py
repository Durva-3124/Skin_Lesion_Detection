"""
tests/test_models.py
Unit tests for model factories, segmentation metrics, and pipeline quality gate.
All tests run without GPU and without dataset files.
"""

import numpy as np
import torch
import pytest

from src.classification.model import get_classification_model
from src.segmentation.model import get_segmentation_model
from src.segmentation.metrics import dice_score, jaccard_score, pixel_accuracy, combined_loss
from src.preprocessing.transforms import remove_hair, get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Classification model factory
# ---------------------------------------------------------------------------

def test_efficientnet_b0_output_shape():
    model = get_classification_model("efficientnet_b0", num_classes=7, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)


def test_swin_small_output_shape():
    model = get_classification_model("swin_small", num_classes=7, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_classification_model("resnet50", num_classes=7)


def test_efficientnet_dropout_inplace_false():
    """MC Dropout requires inplace=False on the dropout layer."""
    model = get_classification_model("efficientnet_b0", num_classes=7, pretrained=False)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            assert not m.inplace, "Dropout must have inplace=False for MC Dropout"


# ---------------------------------------------------------------------------
# Segmentation model factory
# ---------------------------------------------------------------------------

def test_unet_vgg16_output_shape():
    model = get_segmentation_model("vgg16", pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 256, 256)


def test_unet_mobilenetv2_output_shape():
    model = get_segmentation_model("mobilenetv2", pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 256, 256)


def test_unknown_encoder_raises():
    with pytest.raises(ValueError, match="Unknown encoder"):
        get_segmentation_model("vgg19")


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def test_dice_perfect():
    logits = torch.ones(2, 1, 4, 4) * 10  # sigmoid → ~1
    target = torch.ones(2, 1, 4, 4)
    assert dice_score(logits, target) > 0.99


def test_dice_zero():
    logits = torch.ones(2, 1, 4, 4) * 10
    target = torch.zeros(2, 1, 4, 4)
    assert dice_score(logits, target) < 0.01


def test_jaccard_perfect():
    logits = torch.ones(2, 1, 4, 4) * 10
    target = torch.ones(2, 1, 4, 4)
    assert jaccard_score(logits, target) > 0.99


def test_pixel_accuracy_perfect():
    logits = torch.ones(2, 1, 4, 4) * 10
    target = torch.ones(2, 1, 4, 4)
    assert pixel_accuracy(logits, target) > 0.99


def test_combined_loss_positive():
    logits = torch.randn(2, 1, 4, 4)
    target = torch.randint(0, 2, (2, 1, 4, 4)).float()
    loss = combined_loss(logits, target)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def test_val_transforms_output_shape():
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    t = get_val_transforms(224)
    out = t(img)
    assert out.shape == (3, 224, 224)


def test_train_transforms_output_shape():
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    t = get_train_transforms(224)
    out = t(img)
    assert out.shape == (3, 224, 224)


def test_remove_hair_returns_same_shape():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out = remove_hair(img)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Quality gate (pipeline)
# ---------------------------------------------------------------------------

def test_quality_gate_rejects_dark():
    from src.pipeline import TejaLensPipeline
    # Instantiate without weights by monkey-patching
    pipe = object.__new__(TejaLensPipeline)
    pipe.class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    dark = np.zeros((224, 224, 3), dtype=np.uint8)
    passed, reason = pipe._quality_check(dark)
    assert not passed
    assert "dark" in reason.lower()


def test_quality_gate_rejects_blurry():
    from src.pipeline import TejaLensPipeline
    pipe = object.__new__(TejaLensPipeline)
    pipe.class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    # Uniform gray = zero Laplacian variance
    blurry = np.full((224, 224, 3), 128, dtype=np.uint8)
    passed, reason = pipe._quality_check(blurry)
    assert not passed
    assert "blurry" in reason.lower()


def test_quality_gate_passes_normal():
    from src.pipeline import TejaLensPipeline
    pipe = object.__new__(TejaLensPipeline)
    pipe.class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    rng = np.random.default_rng(0)
    normal = rng.integers(50, 200, (224, 224, 3), dtype=np.uint8)
    passed, _ = pipe._quality_check(normal)
    assert passed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_paths_are_pathlib():
    import pathlib
    from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
    for p in (DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR):
        assert isinstance(p, pathlib.Path)
