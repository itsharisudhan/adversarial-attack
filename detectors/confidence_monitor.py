"""
Confidence Monitor Module
Detects adversarial examples based on confidence score and entropy analysis
"""

import torch
import torch.nn.functional as F
import numpy as np


class ConfidenceMonitor:
    """
    Monitors model confidence scores and entropy to detect adversarial examples.
    
    Complexity: O(k) where k is number of classes
    """
    
    def __init__(self, entropy_threshold=0.5, confidence_threshold=0.95):
        """
        Initialize the confidence monitor.
        
        Args:
            entropy_threshold (float): Threshold for entropy-based detection
            confidence_threshold (float): Threshold for high confidence detection
        """
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.detection_count = 0
        self.total_count = 0
        
    def calculate_entropy(self, probs):
        """
        Calculate Shannon entropy of probability distribution.
        
        Formula: H(p) = -Σ p(i) * log(p(i))
        
        Args:
            probs: Probability distribution tensor
            
        Returns:
            float: Entropy value
        """
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)
            
        # Ensure probabilities sum to 1
        probs = probs / (torch.sum(probs) + 1e-8)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        return entropy.item()
    
    def calculate_max_confidence(self, probs):
        """
        Get maximum confidence score.
        
        Args:
            probs: Probability distribution
            
        Returns:
            float: Maximum probability
        """
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs)
            
        return torch.max(probs).item()
    
    def detect_anomaly(self, model_output, return_probs=False):
        """
        Detect anomaly based on confidence and entropy patterns.
        
        Adversarial examples often have:
        - Very low entropy (overly confident)
        - Very high maximum confidence
        
        Args:
            model_output: Raw model output (logits or probabilities)
            return_probs: Whether to return probability distribution
            
        Returns:
            dict: Detection result
        """
        self.total_count += 1
        
        # Convert to probabilities if needed
        if isinstance(model_output, np.ndarray):
            model_output = torch.from_numpy(model_output)
            
        # Apply softmax if values seem like logits
        if torch.max(model_output) > 1.0 or torch.min(model_output) < 0:
            probs = F.softmax(model_output, dim=-1)
        else:
            probs = model_output
            
        # Calculate metrics
        entropy = self.calculate_entropy(probs)
        max_conf = self.calculate_max_confidence(probs)
        
        # Detection logic: flag if entropy too low AND confidence too high
        is_adversarial = (entropy < self.entropy_threshold and 
                         max_conf > self.confidence_threshold)
        
        if is_adversarial:
            self.detection_count += 1
        
        anomaly = (1 - entropy) * max_conf  # Combined score
        if anomaly < 0:
            anomaly = 0.0
        result = {
            'is_adversarial': is_adversarial,
            'entropy': entropy,
            'max_confidence': max_conf,
            'entropy_threshold': self.entropy_threshold,
            'confidence_threshold': self.confidence_threshold,
            'anomaly_score': anomaly
        }
        
        if return_probs:
            result['probabilities'] = probs
            
        return result
    
    def detect_batch(self, model_outputs):
        """
        Detect anomalies in a batch of outputs.
        
        Args:
            model_outputs: Batch of model outputs
            
        Returns:
            list: List of detection results
        """
        results = []
        for output in model_outputs:
            results.append(self.detect_anomaly(output))
        return results
    
    def get_statistics(self):
        """Get detection statistics."""
        return {
            'total_processed': self.total_count,
            'anomalies_detected': self.detection_count,
            'detection_rate': self.detection_count / max(self.total_count, 1)
        }
    
    def reset_statistics(self):
        """Reset detection counters."""
        self.detection_count = 0
        self.total_count = 0
