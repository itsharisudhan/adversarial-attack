"""
LID + Mahalanobis Adversarial Detector

Detects adversarial perturbations by measuring how far an image embedding
sits from the manifold of clean images.  **No training on adversarial
examples is required** — this is fully unsupervised.

Algorithms
----------
1. Local Intrinsic Dimensionality (LID)
   Measures the intrinsic dimensionality around a query point.
   Clean images → low LID (on-manifold).
   Adversarial images → high LID (off-manifold).

2. Mahalanobis Distance
   Measures how many standard deviations away the point is from the
   centroid of the clean distribution, accounting for feature correlations.

References
----------
- Ma et al., ICLR 2018 – LID for adversarial detection
- Lee et al., NeurIPS 2018 – Mahalanobis distance for OOD/adversarial
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing faiss; fall back gracefully
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore[import-untyped]
except ModuleNotFoundError:
    faiss = None  # type: ignore[assignment]
    LOGGER.info("faiss not installed — LID detector will use brute-force k-NN")


def _brute_force_knn(query: np.ndarray, reference: np.ndarray, k: int):
    """Fallback k-NN when FAISS is not installed."""
    diff = reference - query[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=1)
    idx = np.argpartition(dists, k)[:k]
    return np.sort(dists[idx])


class LIDDetector:
    """
    Adversarial detector using Local Intrinsic Dimensionality
    and Mahalanobis distance on feature embeddings.

    Parameters
    ----------
    k : int
        Number of nearest neighbours for LID computation.
    lid_threshold : float
        LID values above this indicate adversarial input.
    maha_threshold : float
        Mahalanobis distances above this indicate adversarial input.
    lid_weight : float
        Weight of LID score in the combined adversarial score.
    maha_weight : float
        Weight of Mahalanobis score in the combined adversarial score.
    """

    def __init__(
        self,
        k: int = 20,
        lid_threshold: float = 60.0,
        maha_threshold: float = 55.0,
        lid_weight: float = 0.6,
        maha_weight: float = 0.4,
    ):
        self.k = k
        self.lid_threshold = lid_threshold
        self.maha_threshold = maha_threshold
        self.lid_weight = lid_weight
        self.maha_weight = maha_weight

        # Populated by load() or build()
        self.reference: np.ndarray | None = None
        self.faiss_index: Any = None
        self.mean: np.ndarray | None = None
        self.cov_inv: np.ndarray | None = None
        self._ready = False

    # ------------------------------------------------------------------
    # Build / Save / Load
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the reference database from clean-image embeddings.

        Parameters
        ----------
        embeddings : (N, D) float32 array
        """
        embeddings = embeddings.astype(np.float32)
        self.reference = embeddings
        self.mean = np.mean(embeddings, axis=0)

        # Regularised covariance inverse (add λI to prevent singularity)
        cov = np.cov(embeddings.T)
        reg = 1e-4 * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.inv(cov + reg)

        # Build FAISS index if available
        if faiss is not None:
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(embeddings)
            LOGGER.info(
                "FAISS index built: %d vectors, dim=%d", len(embeddings), dim
            )
        else:
            LOGGER.info(
                "FAISS unavailable — using brute-force k-NN (%d vectors)",
                len(embeddings),
            )

        self._ready = True

    def save(self, directory: str | Path) -> None:
        """Save reference DB, FAISS index, and statistics to disk."""
        import shutil
        import tempfile

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.reference is not None:
            np.save(directory / "reference_embeddings.npy", self.reference)
        np.savez(
            directory / "reference_stats.npz",
            mean=self.mean,
            cov_inv=self.cov_inv,
        )
        if faiss is not None and self.faiss_index is not None:
            target = directory / "reference.faiss"
            try:
                faiss.write_index(self.faiss_index, str(target))
            except RuntimeError:
                # FAISS C++ can't handle Unicode paths — write to temp, then copy
                with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                    tmp_path = tmp.name
                faiss.write_index(self.faiss_index, tmp_path)
                shutil.copy2(tmp_path, str(target))
                os.remove(tmp_path)

        LOGGER.info("Reference DB saved to %s", directory)

    def load(self, directory: str | Path) -> bool:
        """Load a previously saved reference DB.  Returns True on success."""
        directory = Path(directory)
        stats_path = directory / "reference_stats.npz"
        embeddings_path = directory / "reference_embeddings.npy"
        faiss_path = directory / "reference.faiss"

        if not stats_path.exists():
            LOGGER.warning("Reference stats not found at %s", stats_path)
            return False

        stats = np.load(stats_path)
        self.mean = stats["mean"]
        self.cov_inv = stats["cov_inv"]

        if embeddings_path.exists():
            self.reference = np.load(embeddings_path)

        if faiss is not None and faiss_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
            except RuntimeError:
                import shutil
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                    tmp_path = tmp.name
                shutil.copy2(str(faiss_path), tmp_path)
                self.faiss_index = faiss.read_index(tmp_path)
                os.remove(tmp_path)
            LOGGER.info(
                "Loaded FAISS index: %d vectors", self.faiss_index.ntotal
            )
        elif self.reference is not None:
            LOGGER.info("Loaded brute-force reference: %d vectors", len(self.reference))

        self._ready = True
        return True

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Detection algorithms
    # ------------------------------------------------------------------

    def compute_lid(self, embedding: np.ndarray) -> float:
        """
        Compute Local Intrinsic Dimensionality for a single embedding.

        LID(x) = [ (1/k) · Σᵢ log(r_max / rᵢ) ]⁻¹

        Low LID → on clean manifold.
        High LID → off manifold (adversarial).
        """
        embedding = embedding.astype(np.float32).reshape(1, -1)
        k = min(self.k, self._num_references() - 1)
        if k < 2:
            return 0.0

        # Get k nearest neighbour distances
        if self.faiss_index is not None:
            distances, _ = self.faiss_index.search(embedding, k + 1)
            # distances are squared L2 from FAISS; take sqrt
            dists = np.sqrt(distances[0][1:])  # skip self-match at index 0
        elif self.reference is not None:
            dists = _brute_force_knn(embedding.flatten(), self.reference, k)
        else:
            return 0.0

        # Avoid zero distances
        dists = np.maximum(dists, 1e-10)
        r_max = dists[-1]

        # MLE estimate of LID
        log_ratios = np.log(r_max / dists[:-1])
        mean_log = np.mean(log_ratios)

        if mean_log < 1e-10:
            return 0.0

        return 1.0 / mean_log

    def compute_mahalanobis(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from the clean distribution.

        D_M(x) = √[ (x - μ)ᵀ · Σ⁻¹ · (x - μ) ]
        """
        if self.mean is None or self.cov_inv is None:
            return 0.0

        diff = embedding.flatten() - self.mean
        return float(np.sqrt(np.clip(diff @ self.cov_inv @ diff, 0, None)))

    def detect(self, embedding: np.ndarray) -> dict:
        """
        Run full adversarial detection on a single embedding.

        Returns
        -------
        dict with: score, lid, mahalanobis, is_adversarial
        """
        if not self._ready:
            return {
                "score": 0.0,
                "lid": 0.0,
                "mahalanobis": 0.0,
                "is_adversarial": False,
                "detail": "Reference DB not loaded",
            }

        lid_score = self.compute_lid(embedding)
        maha_score = self.compute_mahalanobis(embedding)

        # Normalise into [0, 1] range
        lid_norm = min(lid_score / (self.lid_threshold * 2), 1.0)
        maha_norm = min(maha_score / (self.maha_threshold * 2), 1.0)

        combined = self.lid_weight * lid_norm + self.maha_weight * maha_norm
        combined = min(1.0, max(0.0, combined))

        return {
            "score": round(float(combined), 4),
            "lid": round(float(lid_score), 4),
            "mahalanobis": round(float(maha_score), 4),
            "is_adversarial": combined > 0.5,
            "lid_threshold": self.lid_threshold,
            "maha_threshold": self.maha_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _num_references(self) -> int:
        if self.faiss_index is not None:
            return self.faiss_index.ntotal
        if self.reference is not None:
            return len(self.reference)
        return 0
