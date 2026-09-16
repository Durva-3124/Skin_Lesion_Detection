"""
src/classification/model.py
Classification models:
  - EfficientNet-B0: on-device deployment candidate
  - Swin-Transformer-Small: research/benchmark model
  - EfficientFormerV2-S2: research/benchmark model (if timm available)
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_efficientnet_b0(num_classes: int, pretrained=True, dropout=0.3) -> nn.Module:
    """
    EfficientNet-B0 — primary on-device deployment candidate.
    Dropout kept as a module attribute so MC Dropout can activate it at inference.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=False),  # inplace=False required for MC Dropout
        nn.Linear(in_features, num_classes),
    )
    return model


def get_swin_small(num_classes: int, pretrained=True) -> nn.Module:
    """
    Swin Transformer Small — research/benchmark model.
    Best performer on harder/imbalanced sets and mobile-acquired images (Module 3).
    """
    weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_s(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def get_efficientformerv2(num_classes: int, pretrained=True) -> nn.Module:
    """
    EfficientFormerV2-S2 — research/benchmark model (requires timm).
    Falls back to EfficientNet-B3 if timm is not installed.
    """
    try:
        import timm
        model = timm.create_model(
            "efficientformerv2_s2",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model
    except ImportError:
        print("timm not installed — falling back to EfficientNet-B3 as research model.")
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model


def get_classification_model(name: str, num_classes: int, pretrained=True) -> nn.Module:
    """
    name: 'efficientnet_b0' | 'swin_small' | 'efficientformerv2'
    """
    if name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, pretrained)
    elif name == "swin_small":
        return get_swin_small(num_classes, pretrained)
    elif name == "efficientformerv2":
        return get_efficientformerv2(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")
