"""
Adversarial Detection Module

Provides both the new unified detection system and legacy detectors.
"""

# ---- New unified detection system ----
from .unified_detector import UnifiedDetector

# ---- Feature-level components (used by unified detector) ----
try:
    from .feature_extractor import FeatureExtractor
    from .lid_detector import LIDDetector
except ImportError:
    FeatureExtractor = None  # type: ignore[assignment,misc]
    LIDDetector = None  # type: ignore[assignment,misc]

# ---- Legacy detectors (still used by demo_60.py / CLI pipeline) ----
from .perturbation_detector import PerturbationDetector
from .confidence_monitor import ConfidenceMonitor
from .activation_analyzer import ActivationAnalyzer
from .detection_system import AdversarialDetectionSystem

__all__ = [
    # New
    'UnifiedDetector',
    'FeatureExtractor',
    'LIDDetector',
    # Legacy
    'PerturbationDetector',
    'ConfidenceMonitor',
    'ActivationAnalyzer',
    'AdversarialDetectionSystem',
]
