"""
Perturbation Detector Module
Detects adversarial examples based on input perturbation analysis using L2 norm
"""

import torch
import numpy as np


class PerturbationDetector:
    """
    Detects adversarial examples by analyzing L2-norm based input deviations.
    
    Complexity: O(n) where n is input dimension
    """
    
    def __init__(self, threshold=0.1, norm_type=2):
        """
        Initialize the perturbation detector.
        
        Args:
            threshold (float): Detection threshold for relative perturbation
            norm_type (int): Type of norm to use (1, 2, or inf)
        """
        self.threshold = threshold
        self.norm_type = norm_type
        self.detection_count = 0
        self.total_count = 0
        
    def calculate_perturbation(self, original_input, test_input):
        """
        Calculate relative perturbation between inputs.
        
        Formula: δ = ||x - x'||₂ / ||x||₂
        
        Args:
            original_input: Original/baseline input tensor
            test_input: Input to be tested
            
        Returns:
            float: Relative perturbation score
        """
        if isinstance(original_input, np.ndarray):
            original_input = torch.from_numpy(original_input)
        if isinstance(test_input, np.ndarray):
            test_input = torch.from_numpy(test_input)
            
        # Calculate norm of difference
        diff = torch.norm(test_input - original_input, p=self.norm_type)
        
        # Calculate norm of original
        orig_norm = torch.norm(original_input, p=self.norm_type)
        
        # Relative difference
        relative_diff = diff / (orig_norm + 1e-8)
        
        return relative_diff.item()
    
    def detect(self, original_input, test_input):
        """
        Detect if test_input is adversarial compared to original_input.
        
        Args:
            original_input: Original/baseline input
            test_input: Input to be tested
            
        Returns:
            dict: Detection result with score and classification
        """
        self.total_count += 1
        
        perturbation_score = self.calculate_perturbation(original_input, test_input)
        is_adversarial = perturbation_score > self.threshold
        
        if is_adversarial:
            self.detection_count += 1
        
        return {
            'is_adversarial': is_adversarial,
            'perturbation_score': perturbation_score,
            'threshold': self.threshold,
            'confidence': min(perturbation_score / self.threshold, 1.0) if is_adversarial else 0.0
        }
    
    def get_statistics(self):
        """Get detection statistics."""
        return {
            'total_processed': self.total_count,
            'adversarial_detected': self.detection_count,
            'detection_rate': self.detection_count / max(self.total_count, 1)
        }
    
    def reset_statistics(self):
        """Reset detection counters."""
        self.detection_count = 0
        self.total_count = 0
