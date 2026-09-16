"""
src/classification/train.py
Trains classification models on HAM10000 or ISIC 2019.
Loss: class-weighted cross-entropy (default) or focal loss.
Records accuracy, F1, sensitivity, specificity per class.
"""

import pathlib
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from src.classification.model import get_classification_model
from src.preprocessing.dataset import get_dataloaders, HAM_CLASSES, ISIC19_CLASSES


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Per-class sensitivity (recall) and specificity
    sensitivity, specificity = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity.append(tp / (tp + fn + 1e-8))
        specificity.append(tn / (tn + fp + 1e-8))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "mean_sensitivity": np.mean(sensitivity),
        "mean_specificity": np.mean(specificity),
        "report": report,
    }


def train(
    model_name="efficientnet_b0",
    dataset_name="ham10000",
    epochs=30,
    lr=1e-4,
    batch_size=32,
    image_size=224,
    loss_type="weighted_ce",   # "weighted_ce" or "focal"
    save_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_name} | Dataset: {dataset_name}")

    class_names = HAM_CLASSES if dataset_name == "ham10000" else ISIC19_CLASSES
    num_classes = len(class_names)

    train_loader, val_loader, class_weights = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size=image_size,
    )
    class_weights = class_weights.to(device)

    model = get_classification_model(model_name, num_classes=num_classes).to(device)

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = save_path or pathlib.Path(f"{model_name}_{dataset_name}.pth")
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        metrics = evaluate(model, val_loader, device, num_classes)
        print(
            f"Epoch {epoch:3d}/{epochs} | loss={train_loss/len(train_loader):.4f} "
            f"| acc={metrics['accuracy']:.4f} | f1={metrics['macro_f1']:.4f} "
            f"| sens={metrics['mean_sensitivity']:.4f} | spec={metrics['mean_specificity']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (f1={best_f1:.4f})")

    print(f"\nBest macro F1: {best_f1:.4f}")
    print("\nFinal classification report:")
    print(metrics["report"])
    return model, metrics


if __name__ == "__main__":
    # Research/benchmark model
    train(model_name="swin_small", dataset_name="ham10000", epochs=30, loss_type="focal")
