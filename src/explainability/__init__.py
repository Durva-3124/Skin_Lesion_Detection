"""
src/explainability/__init__.py
Grad-CAM heatmap generation for EfficientNet-B0.
Targets the last convolutional block (features[-1]) via forward/backward hooks.
"""

import numpy as np
import torch
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks the last conv block in model.features to capture activations and gradients.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a heatmap (H x W, float32, 0–1) for the given class index.
        tensor: preprocessed image tensor (C x H x W), no batch dim.
        """
        self.model.eval()
        inp = tensor.unsqueeze(0).to(self.device)
        inp.requires_grad_(False)

        self.model.zero_grad()
        logits = self.model(inp)
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


def overlay_heatmap(original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blends a Grad-CAM heatmap onto the original RGB image.
    original_image: H x W x 3, uint8.
    cam: H x W, float32, 0–1.
    Returns: H x W x 3, uint8.
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (original_image * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
