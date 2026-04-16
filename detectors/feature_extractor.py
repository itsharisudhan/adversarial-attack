"""
Shared Feature Extractor — EfficientNet-B0

Produces a 1280-dimensional embedding for any input image.
Uses ImageNet-pretrained weights (no custom training required).
"""

import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

LOGGER = logging.getLogger(__name__)

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
EMBEDDING_DIM = 1280


class FeatureExtractor:
    """
    Extract 1280-d feature embeddings from images using EfficientNet-B0.

    The classification head is removed — only the convolutional backbone
    and global average pool are kept.  Pretrained ImageNet weights are
    loaded automatically on first use.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model: nn.Module | None = None

        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Lazy model loading — avoids slow startup if detector isn't used
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _build_model(self) -> nn.Module:
        """Load EfficientNet-B0 and strip the classification head."""
        LOGGER.info("Loading EfficientNet-B0 (pretrained=ImageNet) …")
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Older torchvision without Weights enum
            base = models.efficientnet_b0(pretrained=True)  # type: ignore[arg-type]

        # Keep everything up to and including the global average pool,
        # but throw away the dropout + linear classifier.
        feature_layers = nn.Sequential(
            base.features,
            base.avgpool,
        )
        feature_layers.eval()
        feature_layers.to(self.device)
        LOGGER.info(
            "EfficientNet-B0 loaded on %s  (output dim = %d)",
            self.device,
            EMBEDDING_DIM,
        )
        return feature_layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert a single PIL image to a normalised (1, 3, 224, 224) tensor."""
        rgb = image.convert("RGB")
        return self.transform(rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract a 1280-d embedding from a single PIL image.

        Returns:
            numpy array of shape (1280,)
        """
        tensor = self._prepare_tensor(image)
        features = self.model(tensor)
        return features.flatten().cpu().numpy()

    @torch.no_grad()
    def extract_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings for a list of PIL images.

        Returns:
            numpy array of shape (N, 1280)
        """
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            tensors = torch.cat(
                [self._prepare_tensor(img) for img in batch_imgs], dim=0
            )
            features = self.model(tensors)
            # Flatten each to (1280,)
            flat = features.view(features.size(0), -1).cpu().numpy()
            all_embeddings.append(flat)
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def extract_from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings from an already-normalised (N, 3, 224, 224) tensor.
        """
        tensor = tensor.to(self.device)
        features = self.model(tensor)
        return features.view(features.size(0), -1).cpu().numpy()
