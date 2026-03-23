"""
Main Adversarial Detection System
Combines multiple detection modules using ensemble approach
"""

import torch
import numpy as np
from .perturbation_detector import PerturbationDetector
from .confidence_monitor import ConfidenceMonitor
from .activation_analyzer import ActivationAnalyzer


class AdversarialDetectionSystem:
    """
    Main detection system combining multiple detection modules.
    Uses weighted ensemble voting for final decision.
    """
    
    def __init__(self, model, weights=None, use_activation_analyzer=False):
        """
        Initialize the detection system.
        
        Args:
            model: PyTorch model to protect
            weights: Dictionary of weights for each detector
            use_activation_analyzer: Whether to use activation analysis (slower)
        """
        self.model = model
        self.use_activation_analyzer = use_activation_analyzer
        
        # Initialize detectors
        self.pert_detector = PerturbationDetector(threshold=0.1)
        self.conf_monitor = ConfidenceMonitor(
            entropy_threshold=0.5, 
            confidence_threshold=0.95
        )
        
        if use_activation_analyzer:
            self.activation_analyzer = ActivationAnalyzer(model, threshold_std=2.0)
        
        # Set weights for ensemble
        if weights is None:
            if use_activation_analyzer:
                self.weights = {
                    'perturbation': 0.35,
                    'confidence': 0.35,
                    'activation': 0.30
                }
            else:
                self.weights = {
                    'perturbation': 0.5,
                    'confidence': 0.5
                }
        else:
            self.weights = weights
            
        # Statistics
        self.total_count = 0
        self.detected_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
    def setup_baseline(self, dataloader, num_batches=10):
        """
        Setup baseline statistics for activation analyzer.
        
        Args:
            dataloader: DataLoader with clean examples
            num_batches: Number of batches to process
        """
        if self.use_activation_analyzer:
            print("Collecting baseline activation statistics...")
            self.activation_analyzer.collect_baseline_statistics(
                dataloader, num_batches
            )
            print("Baseline collection complete.")
    
    def detect(self, input_data, baseline_input=None, ground_truth=None):
        """
        Detect if input is adversarial using ensemble approach.
        
        Args:
            input_data: Input tensor to test
            baseline_input: Original clean input for perturbation analysis
            ground_truth: True label (0=clean, 1=adversarial) for evaluation
            
        Returns:
            dict: Detection results with scores and decision
        """
        self.total_count += 1
        
        # Get model output
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        
        # Run detectors
        detection_scores = {}
        
        # 1. Perturbation detection
        if baseline_input is not None:
            pert_result = self.pert_detector.detect(baseline_input, input_data)
            detection_scores['perturbation'] = float(pert_result['is_adversarial'])
            pert_score = pert_result['perturbation_score']
        else:
            detection_scores['perturbation'] = 0.0
            pert_score = 0.0
        
        # 2. Confidence monitoring
        conf_result = self.conf_monitor.detect_anomaly(output)
        detection_scores['confidence'] = float(conf_result['is_adversarial'])
        conf_anomaly_score = conf_result['anomaly_score']
        
        # 3. Activation analysis (if enabled)
        if self.use_activation_analyzer:
            act_result = self.activation_analyzer.analyze_activation_pattern(input_data)
            detection_scores['activation'] = float(act_result['is_adversarial'])
            act_deviation = act_result['average_deviation']
        else:
            act_deviation = 0.0
        
        # Weighted ensemble decision
        final_score = sum(
            self.weights[key] * detection_scores[key] 
            for key in detection_scores.keys()
        )
        # use >= so that a perfect perturbation vote alone (0.5 when two detectors)
        # is sufficient to trigger; avoids 0.5 tie being considered clean.
        is_adversarial = final_score >= 0.5
        
        if is_adversarial:
            self.detected_count += 1
        
        # Update confusion matrix if ground truth provided
        if ground_truth is not None:
            if ground_truth == 1 and is_adversarial:
                self.true_positives += 1
            elif ground_truth == 0 and is_adversarial:
                self.false_positives += 1
            elif ground_truth == 0 and not is_adversarial:
                self.true_negatives += 1
            elif ground_truth == 1 and not is_adversarial:
                self.false_negatives += 1
        
        return {
            'is_adversarial': is_adversarial,
            'confidence': final_score,
            'detector_scores': detection_scores,
            'weights': self.weights,
            'details': {
                'perturbation_score': pert_score,
                'confidence_anomaly': conf_anomaly_score,
                'activation_deviation': act_deviation,
                'model_prediction': torch.argmax(output).item()
            }
        }
    
    def detect_batch(self, batch_data, baseline_batch=None, ground_truths=None):
        """
        Detect adversarial examples in a batch.
        
        Args:
            batch_data: Batch of inputs to test
            baseline_batch: Batch of original clean inputs
            ground_truths: Ground truth labels for evaluation
            
        Returns:
            list: List of detection results
        """
        results = []
        
        for i in range(len(batch_data)):
            baseline = baseline_batch[i] if baseline_batch is not None else None
            gt = ground_truths[i] if ground_truths is not None else None
            
            result = self.detect(
                batch_data[i:i+1], 
                baseline.unsqueeze(0) if baseline is not None else None,
                gt
            )
            results.append(result)
        
        return results
    
    def evaluate(self):
        """
        Calculate evaluation metrics.
        
        Returns:
            dict: Metrics including accuracy, precision, recall, F1
        """
        total = self.true_positives + self.false_positives + \
                self.true_negatives + self.false_negatives
        
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive_rate': 0.0
            }
        
        accuracy = (self.true_positives + self.true_negatives) / total
        
        precision = self.true_positives / (self.true_positives + self.false_positives) \
                   if (self.true_positives + self.false_positives) > 0 else 0.0
        
        recall = self.true_positives / (self.true_positives + self.false_negatives) \
                if (self.true_positives + self.false_negatives) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) \
                  if (precision + recall) > 0 else 0.0
        
        fpr = self.false_positives / (self.false_positives + self.true_negatives) \
             if (self.false_positives + self.true_negatives) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'total_processed': total
        }
    
    def get_statistics(self):
        """Get overall detection statistics."""
        return {
            'total_processed': self.total_count,
            'adversarial_detected': self.detected_count,
            'detection_rate': self.detected_count / max(self.total_count, 1),
            'detector_stats': {
                'perturbation': self.pert_detector.get_statistics(),
                'confidence': self.conf_monitor.get_statistics()
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_count = 0
        self.detected_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.pert_detector.reset_statistics()
        self.conf_monitor.reset_statistics()
        if self.use_activation_analyzer:
            self.activation_analyzer.reset_statistics()
