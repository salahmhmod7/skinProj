"""Shared model loading + preprocessing for HAM10000 ConvNeXt-Base.

Extracted from Untitled-1.ipynb (cells 36/38): ConvNeXt-Base, 384x384,
custom classifier head, melanoma-thresholded decision rule.
"""
import os

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import models

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / Bowen's disease (pre-cancerous)",
    "bcc": "Basal cell carcinoma (malignant)",
    "bkl": "Benign keratosis-like lesions (benign)",
    "df": "Dermatofibroma (benign)",
    "mel": "Melanoma (malignant - high risk)",
    "nv": "Melanocytic nevi (benign mole)",
    "vasc": "Vascular lesions (benign)",
}
NUM_CLASSES = 7
MEL_IDX = CLASS_NAMES.index("mel")

IMG_SIZE = int(os.getenv("IMG_SIZE", "384"))
MEL_THRESHOLD = float(os.getenv("MEL_THRESHOLD", "0.30"))
MODEL_PATH = os.getenv("MODEL_PATH", "data/convnext_clinical_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm matching the training checkpoint keys."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


def build_model() -> nn.Module:
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features  # 1024 for Base
    model.classifier = nn.Sequential(
        LayerNorm2d(in_features),
        nn.Flatten(start_dim=1),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def load_model(ckpt_path: str = MODEL_PATH, device: torch.device = DEVICE) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Mount it with `-v ./data:/app/data:ro` or set MODEL_PATH."
        )
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_transform(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def preprocess_image(image_path: str, img_size: int = IMG_SIZE,
                     device: torch.device = DEVICE) -> torch.Tensor:
    transform = get_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(image=np.array(img))["image"]
    return tensor.unsqueeze(0).to(device)


def predict_image(image_path: str, model: nn.Module,
                  apply_threshold: bool = True,
                  mel_threshold: float = MEL_THRESHOLD,
                  img_size: int = IMG_SIZE,
                  device: torch.device = DEVICE) -> dict:
    tensor = preprocess_image(image_path, img_size, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    mel_prob = float(probs[MEL_IDX])
    if apply_threshold and mel_prob >= mel_threshold:
        pred_idx = MEL_IDX
    else:
        masked = probs.copy()
        if apply_threshold:
            masked[MEL_IDX] = -1.0
        pred_idx = int(np.argmax(masked))

    pred_label = CLASS_NAMES[pred_idx]
    return {
        "pred_label": pred_label,
        "description": CLASS_DESCRIPTIONS[pred_label],
        "confidence": float(probs[pred_idx] * 100.0),
        "mel_probability": mel_prob * 100.0,
        "mel_threshold": mel_threshold * 100.0,
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i] * 100.0) for i in range(NUM_CLASSES)},
    }
