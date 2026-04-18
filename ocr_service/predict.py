"""
Inference logic for the OCR pipeline.

Loads CharClassifier weights, runs character patches through the model,
and maps predicted indices back to the EMNIST character set.
"""

import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_service.model import CharClassifier

# EMNIST byclass label mapping: 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z
_LABELS = (
    [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)

_classifier: Optional[CharClassifier] = None
_clf_device: str = "cpu"
_to_tensor = T.ToTensor()


def load_classifier(weights_path: str, device: str = "cpu") -> None:
    """Load CharClassifier weights into module-level singleton."""
    global _classifier, _clf_device
    model = CharClassifier(num_classes=62)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _classifier = model.to(device)
    _clf_device = device


def predict_patches(patches: List[np.ndarray], batch_size: int = 64) -> List[str]:
    """
    Run CharClassifier over a list of 32×32 uint8 grayscale patches.
    Returns a list of predicted characters.
    """
    if _classifier is None:
        raise RuntimeError("Classifier not loaded. Call load_classifier() first.")
    if not patches:
        return []

    # Convert all patches to tensors
    tensors = []
    for patch in patches:
        img = Image.fromarray(patch, mode="L")
        tensors.append(_to_tensor(img))  # (1, 32, 32)

    results = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(_clf_device)
        with torch.no_grad():
            logits = _classifier(batch)
            preds = logits.argmax(dim=1).cpu().tolist()
        results.extend([_LABELS[p] for p in preds])

    return results


def predict_image(
    image_source,
    denoise: bool = True,
    patch_size: int = 32,
    batch_size: int = 64,
) -> dict:
    """
    Full OCR inference on a single image.

    Returns:
        {
          "characters": ["H", "e", "l", "l", "o", ...],
          "text": "Hello...",
          "num_chars": 5,
        }
    """
    from ocr_service.preprocess import preprocess_image

    patches = preprocess_image(image_source, denoise=denoise, patch_size=patch_size)
    chars = predict_patches(patches, batch_size=batch_size)

    return {
        "characters": chars,
        "text": "".join(chars),
        "num_chars": len(chars),
    }
