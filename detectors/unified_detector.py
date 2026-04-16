"""
Unified Fraud Detection System

Orchestrates three detection layers into one verdict:
  1. Adversarial Head — LID + Mahalanobis (gradient attack detection)
  2. Forensic Head   — FFT + ELA + Stats + Texture (AI-image detection)
  3. Duplicate Head  — SHA-256 + cosine similarity (reused-photo detection)

Threat types detected:
  • CLEAN              — legitimate image
  • AI_GENERATED       — fully synthetic (Stable Diffusion, DALL-E, etc.)
  • ADVERSARIAL_ATTACK — invisible perturbation (FGSM, PGD, etc.)
  • TAMPERED           — real image that has been edited / modified
  • HYBRID_THREAT      — adversarial + synthetic combined
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Default directory for checkpoints (reference DB, model weights)
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# ---------------------------------------------------------------------------
# Try importing torch-based components; fail gracefully if not available
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    from .feature_extractor import FeatureExtractor
    from .lid_detector import LIDDetector

    _TORCH_AVAILABLE = True
except Exception as exc:
    LOGGER.info("PyTorch components unavailable: %s — forensic-only mode", exc)


class UnifiedDetector:
    """
    All-in-one detector for food-delivery fraud:
    adversarial perturbations, AI-generated images, edited images,
    and duplicate submissions.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str | None = None,
        enable_duplicate_check: bool = True,
        duplicate_history_size: int = 500,
    ):
        self.checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self.device = device

        # ---- Adversarial head (LID + Mahalanobis) ----
        self._feature_extractor: Any = None
        self._lid_detector: Any = None
        self._adv_ready = False

        # ---- Duplicate detection ----
        self._enable_duplicate = enable_duplicate_check
        self._hash_history: deque[str] = deque(maxlen=duplicate_history_size)
        self._embedding_history: deque[np.ndarray] = deque(
            maxlen=duplicate_history_size
        )

        # ---- Try loading models ----
        self._try_init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _try_init(self) -> None:
        """Attempt to load the embedding model and reference database."""
        if not _TORCH_AVAILABLE:
            LOGGER.info("Running in forensic-only mode (no PyTorch)")
            return

        try:
            self._feature_extractor = FeatureExtractor(device=self.device)
            self._lid_detector = LIDDetector()
            loaded = self._lid_detector.load(self.checkpoint_dir)
            if loaded:
                self._adv_ready = True
                LOGGER.info("Unified detector: adversarial head READY")
            else:
                LOGGER.warning(
                    "Reference DB not found in %s — adversarial head disabled. "
                    "Run  python data/build_reference_db.py  to build it.",
                    self.checkpoint_dir,
                )
        except Exception as exc:
            LOGGER.warning("Failed to initialise adversarial head: %s", exc)

    @property
    def adversarial_head_ready(self) -> bool:
        return self._adv_ready

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _check_duplicate(
        self, image_bytes: bytes | None, embedding: np.ndarray | None
    ) -> dict:
        """Check for exact or near-duplicate submissions."""
        result: dict[str, Any] = {
            "is_duplicate": False,
            "hash_match": False,
            "similarity_match": False,
            "max_similarity": 0.0,
        }
        if not self._enable_duplicate:
            return result

        # Exact hash match
        if image_bytes is not None:
            img_hash = hashlib.sha256(image_bytes).hexdigest()
            if img_hash in self._hash_history:
                result["is_duplicate"] = True
                result["hash_match"] = True
            self._hash_history.append(img_hash)

        # Near-duplicate via cosine similarity
        if embedding is not None and len(self._embedding_history) > 0:
            query = embedding / (np.linalg.norm(embedding) + 1e-10)
            best_sim = 0.0
            for stored in self._embedding_history:
                ref = stored / (np.linalg.norm(stored) + 1e-10)
                sim = float(np.dot(query, ref))
                best_sim = max(best_sim, sim)
            result["max_similarity"] = round(best_sim, 4)
            if best_sim > 0.97:
                result["is_duplicate"] = True
                result["similarity_match"] = True
            self._embedding_history.append(embedding.copy())

        return result

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image_pil: Image.Image,
        image_bytes: bytes | None = None,
        forensic_result: dict | None = None,
    ) -> dict:
        """
        Run the full detection pipeline on a single image.

        Parameters
        ----------
        image_pil : PIL.Image
            The input image.
        image_bytes : bytes, optional
            Raw file bytes (for hash-based duplicate check).
        forensic_result : dict, optional
            Pre-computed forensic analysis result from web/app.py.
            If None, forensic scores are not included.

        Returns
        -------
        dict
            Comprehensive detection result with verdict, scores, and details.
        """
        result: dict[str, Any] = {
            "verdict": "CLEAN",
            "confidence": 0.0,
            "adversarial_score": 0.0,
            "genai_score": 0.0,
            "is_duplicate": False,
            "details": {},
        }

        # ---- Embedding extraction + adversarial head ----
        embedding = None
        if self._adv_ready and self._feature_extractor is not None:
            try:
                embedding = self._feature_extractor.extract(image_pil)
                adv_result = self._lid_detector.detect(embedding)
                result["adversarial_score"] = adv_result["score"]
                result["details"]["lid_score"] = adv_result["lid"]
                result["details"]["mahalanobis_distance"] = adv_result[
                    "mahalanobis"
                ]
                result["details"]["lid_threshold"] = adv_result["lid_threshold"]
            except Exception as exc:
                LOGGER.warning("Adversarial head error: %s", exc)

        # ---- Forensic head (scores passed in from web/app.py) ----
        if forensic_result is not None:
            genai_score = forensic_result.get("ensemble_score", 0.0)
            result["genai_score"] = genai_score
            result["details"]["forensic_detectors"] = {
                k: v.get("score", 0.0)
                for k, v in forensic_result.get("detectors", {}).items()
            }

        # ---- Duplicate check ----
        dup_result = self._check_duplicate(image_bytes, embedding)
        result["is_duplicate"] = dup_result["is_duplicate"]
        result["details"]["duplicate"] = dup_result

        # ---- Decision fusion ----
        verdict, confidence = self._fuse(
            adv_score=result["adversarial_score"],
            genai_score=result["genai_score"],
            is_duplicate=dup_result["is_duplicate"],
            forensic_result=forensic_result,
        )
        result["verdict"] = verdict
        result["confidence"] = round(confidence, 4)

        return result

    # ------------------------------------------------------------------
    # Decision fusion logic
    # ------------------------------------------------------------------

    def _fuse(
        self,
        adv_score: float,
        genai_score: float,
        is_duplicate: bool,
        forensic_result: dict | None,
    ) -> tuple[str, float]:
        """
        Combine all detection signals into a single verdict.

        Key calibration insight:
          - Clean images:     LID 17-46,  adv_score 0.19-0.39
          - FGSM adversarial: LID 60-80,  adv_score 0.55-0.65
          - AI-generated:     LID 100-170, adv_score 0.80-0.92
          AI images have EXTREME adv_scores because they sit very
          far from the natural-image manifold.  The distinction is
          that adversarial perturbations show moderate deviation
          while AI generation shows extreme deviation.
        """

        # 1. Duplicate — highest priority
        if is_duplicate:
            return "DUPLICATE_FRAUD", 0.98

        # Count forensic detectors flagging above 0.5
        forensic_flags = 0
        ela_score = 0.0
        if forensic_result and "detectors" in forensic_result:
            for det_name, det_data in forensic_result["detectors"].items():
                score = det_data.get("score", 0.0)
                if score > 0.5:
                    forensic_flags += 1
                if det_name == "ela":
                    ela_score = score

        # 2. Extreme adv_score (>0.85) = almost certainly AI-generated
        #    Real FGSM attacks cause adv_score ~0.55-0.65, not 0.85+
        if adv_score > 0.85:
            if genai_score >= 0.45:
                return "HYBRID_THREAT", (adv_score + genai_score) / 2
            else:
                # Very high manifold deviation + low forensic = AI-generated
                # (forensic might miss it but manifold doesn't lie)
                return "AI_GENERATED", adv_score

        # 3. High adv_score (0.70-0.85) with forensic agreement = hybrid
        if adv_score > 0.70 and genai_score >= 0.45:
            return "HYBRID_THREAT", (adv_score + genai_score) / 2

        # 4. High adv_score (0.70-0.85) alone, low forensic = adversarial
        if adv_score > 0.70 and genai_score < 0.45:
            return "ADVERSARIAL_ATTACK", adv_score

        # 5. Strong forensic (genai >= 0.60) with multi-detector agreement
        if genai_score >= 0.60 and forensic_flags >= 2:
            return "AI_GENERATED", genai_score

        # 6. Moderate adv_score (0.55-0.70) + forensic evidence = tampered
        if adv_score > 0.55 and genai_score >= 0.40 and forensic_flags >= 2:
            return "TAMPERED", (adv_score + genai_score) / 2

        # 7. Both adv and genai moderately elevated = suspicious (catches gap)
        #    ai_landscape: adv=0.578, genai=0.512 — falls here
        if adv_score > 0.50 and genai_score >= 0.40:
            combined = (adv_score + genai_score) / 2
            if combined > 0.48:
                return "AI_GENERATED", combined

        # 8. Moderate forensic with strong multi-detector consensus (3+ flags)
        if genai_score >= 0.50 and forensic_flags >= 3:
            return "AI_GENERATED", genai_score

        # 9. Moderate ELA-driven tampering signal
        if genai_score >= 0.45 and ela_score > 0.55 and forensic_flags >= 2:
            return "TAMPERED", genai_score

        # 10. Moderate adversarial signal alone
        if adv_score > 0.65:
            return "ADVERSARIAL_ATTACK", adv_score

        # 11. Clean
        clean_confidence = 1.0 - max(adv_score, genai_score)
        return "CLEAN", max(0.0, clean_confidence)

