"""
Activation Analyzer Module
Analyzes neural network layer activations to detect adversarial examples
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class ActivationAnalyzer:
    """
    Analyzes activation patterns across neural network layers.
    
    Complexity: O(L×N) where L is layers, N is neurons per layer
    """
    
    def __init__(self, model, threshold_std=2.0):
        """
        Initialize the activation analyzer.
        
        Args:
            model: PyTorch model to analyze
            threshold_std: Standard deviation threshold for anomaly detection
        """
        self.model = model
        self.threshold_std = threshold_std
        self.activation_stats = defaultdict(dict)
        self.hooks = []
        self.activations = {}
        self.detection_count = 0
        self.total_count = 0
        
    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def collect_baseline_statistics(self, dataloader, num_batches=10):
        """
        Collect baseline activation statistics from clean data.
        
        Args:
            dataloader: DataLoader with clean examples
            num_batches: Number of batches to process
        """
        self.register_hooks()
        self.model.eval()
        
        activation_values = defaultdict(list)
        
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                # Forward pass
                _ = self.model(data)
                
                # Collect activations
                for name, activation in self.activations.items():
                    activation_values[name].append(activation.flatten())
        
        # Calculate statistics
        for name, values in activation_values.items():
            all_values = torch.cat(values)
            self.activation_stats[name] = {
                'mean': torch.mean(all_values).item(),
                'std': torch.std(all_values).item(),
                'min': torch.min(all_values).item(),
                'max': torch.max(all_values).item()
            }
        
        self.remove_hooks()
    
    def analyze_activation_pattern(self, input_data):
        """
        Analyze activation pattern for given input.
        
        Args:
            input_data: Input tensor to analyze
            
        Returns:
            dict: Analysis results
        """
        self.total_count += 1
        self.register_hooks()
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_data)
        
        anomaly_scores = []
        layer_anomalies = {}
        
        for name, activation in self.activations.items():
            if name not in self.activation_stats:
                continue
                
            stats = self.activation_stats[name]
            act_flat = activation.flatten()
            
            # Calculate deviation from baseline
            mean_activation = torch.mean(act_flat).item()
            deviation = abs(mean_activation - stats['mean']) / (stats['std'] + 1e-8)
            
            is_anomalous = deviation > self.threshold_std
            anomaly_scores.append(deviation)
            layer_anomalies[name] = {
                'deviation': deviation,
                'is_anomalous': is_anomalous,
                'current_mean': mean_activation,
                'baseline_mean': stats['mean']
            }
        
        self.remove_hooks()
        
        # Overall anomaly decision
        avg_deviation = np.mean(anomaly_scores) if anomaly_scores else 0
        is_adversarial = avg_deviation > self.threshold_std
        
        if is_adversarial:
            self.detection_count += 1
        
        return {
            'is_adversarial': is_adversarial,
            'average_deviation': avg_deviation,
            'max_deviation': max(anomaly_scores) if anomaly_scores else 0,
            'num_anomalous_layers': sum(1 for scores in layer_anomalies.values() 
                                       if scores['is_anomalous']),
            'layer_details': layer_anomalies
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
