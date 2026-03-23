"""
Advanced Adversarial Attacks
Implements DeepFool and Boundary Attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdvancedAttacks:
    """Advanced adversarial attack methods (60% milestone)."""
    
    @staticmethod
    def deepfool_attack(model, images, max_iter=50, overshoot=0.02):
        """
        DeepFool attack - finds minimum perturbation to change classification.
        Simplified implementation for demonstration.
        
        Args:
            model: Target model
            images: Input images
            max_iter: Maximum iterations
            overshoot: Overshoot parameter
            
        Returns:
            Adversarial examples
        """
        model.eval()
        device = next(model.parameters()).device
        images = images.to(device)
        
        perturbed = images.clone()
        
        for img_idx in range(images.shape[0]):
            x = images[img_idx:img_idx+1].clone().requires_grad_(True)
            
            # Get original prediction
            output = model(x)
            orig_label = output.argmax(dim=1)
            
            # Iteratively find minimum perturbation
            for _ in range(max_iter):
                output = model(x)
                pred_label = output.argmax(dim=1)
                
                # If prediction changed, stop
                if pred_label != orig_label:
                    break
                
                # Calculate gradients for all classes
                model.zero_grad()
                output[0, orig_label].backward(retain_graph=True)
                grad_orig = x.grad.data.clone()
                
                # Find closest decision boundary
                min_dist = float('inf')
                best_grad = None
                
                for k in range(output.shape[1]):
                    if k == orig_label:
                        continue
                    
                    x.grad.zero_()
                    output[0, k].backward(retain_graph=True)
                    grad_k = x.grad.data.clone()
                    
                    w_k = grad_k - grad_orig
                    f_k = output[0, k] - output[0, orig_label]
                    
                    dist = abs(f_k.item()) / (torch.norm(w_k).item() + 1e-8)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_grad = w_k
                
                # Update perturbation
                if best_grad is not None:
                    r = (min_dist + 1e-4) * best_grad / (torch.norm(best_grad) + 1e-8)
                    x = x + r * (1 + overshoot)
                    x = torch.clamp(x, 0, 1)
                    x = x.detach().requires_grad_(True)
            
            perturbed[img_idx:img_idx+1] = x.detach()
        
        return perturbed
    
    @staticmethod
    def boundary_attack(model, images, labels, max_iter=1000, delta=0.01):
        """
        Boundary Attack - decision-based attack (doesn't need gradients).
        Simplified implementation for demonstration.
        
        Args:
            model: Target model
            images: Input images
            labels: True labels
            max_iter: Maximum iterations
            delta: Step size
            
        Returns:
            Adversarial examples
        """
        model.eval()
        device = next(model.parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        
        # Start from random noise
        perturbed = torch.rand_like(images)
        
        # Ensure it's misclassified
        with torch.no_grad():
            pred = model(perturbed).argmax(dim=1)
            mask = (pred == labels)
            
            # If still classified correctly, add more noise
            while mask.any():
                perturbed[mask] = torch.rand_like(perturbed[mask])
                pred = model(perturbed).argmax(dim=1)
                mask = (pred == labels)
        
        # Iteratively move towards original image
        for _ in range(max_iter):
            # Random perturbation
            noise = torch.randn_like(perturbed) * delta
            candidate = perturbed + noise
            candidate = torch.clamp(candidate, 0, 1)
            
            # Check if still misclassified
            with torch.no_grad():
                pred = model(candidate).argmax(dim=1)
                success = (pred != labels)
                
                # Update if successful and closer to original
                dist_old = torch.norm((perturbed - images).view(images.shape[0], -1), dim=1)
                dist_new = torch.norm((candidate - images).view(images.shape[0], -1), dim=1)
                
                improve = success & (dist_new < dist_old)
                perturbed[improve] = candidate[improve]
        
        return perturbed


def test_advanced_attacks(model, dataloader, device='cpu'):
    """
    Test advanced attacks and measure success rates.
    
    Args:
        model: Target model
        dataloader: Test data loader
        device: Device to use
        
    Returns:
        dict: Attack success rates
    """
    model.eval()
    model = model.to(device)
    
    results = {
        'deepfool': {'total': 0, 'successful': 0},
        'boundary': {'total': 0, 'successful': 0}
    }
    
    # Test on small subset
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Original predictions
        with torch.no_grad():
            orig_pred = model(images).argmax(dim=1)
            correct_mask = (orig_pred == labels)
        
        # Only test on correctly classified samples
        if not correct_mask.any():
            continue
        
        images = images[correct_mask]
        labels = labels[correct_mask]
        
        # DeepFool attack
        try:
            adv_deepfool = AdvancedAttacks.deepfool_attack(model, images, max_iter=10)
            with torch.no_grad():
                pred_deepfool = model(adv_deepfool).argmax(dim=1)
                success_deepfool = (pred_deepfool != labels).sum().item()
            
            results['deepfool']['total'] += len(labels)
            results['deepfool']['successful'] += success_deepfool
        except:
            pass
        
        # Boundary attack (on smaller batch for speed)
        try:
            batch_size = min(5, len(images))
            adv_boundary = AdvancedAttacks.boundary_attack(
                model, images[:batch_size], labels[:batch_size], max_iter=100
            )
            with torch.no_grad():
                pred_boundary = model(adv_boundary).argmax(dim=1)
                success_boundary = (pred_boundary != labels[:batch_size]).sum().item()
            
            results['boundary']['total'] += batch_size
            results['boundary']['successful'] += success_boundary
        except:
            pass
        
        # Test only first batch for speed
        break
    
    # Calculate success rates
    for attack in results:
        total = results[attack]['total']
        successful = results[attack]['successful']
        results[attack]['success_rate'] = (successful / total * 100) if total > 0 else 0
    
    return results
