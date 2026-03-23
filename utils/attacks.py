"""
Adversarial Attack Generation
Implements FGSM, PGD, and C&W attacks for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialAttacks:
    """Collection of adversarial attack methods."""

    @staticmethod
    def _prepare_bound(value, reference_tensor):
        """Broadcast scalar or per-channel bounds to match the input tensor."""
        if value is None:
            return None

        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=reference_tensor.dtype, device=reference_tensor.device)
        else:
            value = value.to(device=reference_tensor.device, dtype=reference_tensor.dtype)

        if value.ndim == 0:
            return value

        while value.ndim < reference_tensor.ndim:
            value = value.unsqueeze(0)

        return value

    @classmethod
    def _clamp_to_bounds(cls, images, clamp_min=None, clamp_max=None):
        """Clamp tensors using scalar or broadcastable tensor bounds."""
        clamp_min = cls._prepare_bound(clamp_min, images)
        clamp_max = cls._prepare_bound(clamp_max, images)

        if clamp_min is not None:
            images = torch.maximum(images, clamp_min)
        if clamp_max is not None:
            images = torch.minimum(images, clamp_max)
        return images
    
    @staticmethod
    def fgsm_attack(model, images, labels, epsilon=0.1, clamp_min=0.0, clamp_max=1.0):
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            model: Target model
            images: Input images
            labels: True labels
            epsilon: Perturbation magnitude
            clamp_min: Minimum allowed value after perturbation
            clamp_max: Maximum allowed value after perturbation
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        data_grad = images.grad.data
        epsilon = AdversarialAttacks._prepare_bound(epsilon, images)
        perturbed_images = images + epsilon * data_grad.sign()
        
        # Clamp to valid range
        perturbed_images = AdversarialAttacks._clamp_to_bounds(
            perturbed_images,
            clamp_min=clamp_min,
            clamp_max=clamp_max
        )
        
        return perturbed_images.detach()
    
    @staticmethod
    def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10,
                   clamp_min=0.0, clamp_max=1.0):
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            model: Target model
            images: Input images
            labels: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
            clamp_min: Minimum allowed value after perturbation
            clamp_max: Maximum allowed value after perturbation
            
        Returns:
            Adversarial examples
        """
        perturbed_images = images.clone().detach()
        epsilon = AdversarialAttacks._prepare_bound(epsilon, images)
        alpha = AdversarialAttacks._prepare_bound(alpha, images)
        
        for _ in range(num_iter):
            perturbed_images.requires_grad = True
            outputs = model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            
            # Update
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = perturbed_images - images
            perturbation = torch.maximum(perturbation, -epsilon)
            perturbation = torch.minimum(perturbation, epsilon)
            perturbed_images = AdversarialAttacks._clamp_to_bounds(
                images + perturbation,
                clamp_min=clamp_min,
                clamp_max=clamp_max
            )
        
        return perturbed_images.detach()
    
    @staticmethod
    def random_noise_attack(images, epsilon=0.1, clamp_min=0.0, clamp_max=1.0):
        """
        Random noise attack (baseline).
        
        Args:
            images: Input images
            epsilon: Noise magnitude
            clamp_min: Minimum allowed value after perturbation
            clamp_max: Maximum allowed value after perturbation
            
        Returns:
            Noisy images
        """
        epsilon = AdversarialAttacks._prepare_bound(epsilon, images)
        noise = torch.randn_like(images) * epsilon
        noisy_images = images + noise
        return AdversarialAttacks._clamp_to_bounds(
            noisy_images,
            clamp_min=clamp_min,
            clamp_max=clamp_max
        )
    
    @staticmethod
    def carlini_wagner_l2(model, images, labels, c=1.0, kappa=0, max_iter=100, 
                          learning_rate=0.01, clamp_min=0.0, clamp_max=1.0):
        """
        Simplified Carlini-Wagner L2 attack.
        
        Args:
            model: Target model
            images: Input images
            labels: True labels
            c: Weight of adversarial loss
            kappa: Confidence parameter
            max_iter: Maximum iterations
            learning_rate: Optimization learning rate
            clamp_min: Minimum allowed value after perturbation
            clamp_max: Maximum allowed value after perturbation
            
        Returns:
            Adversarial examples
        """
        batch_size = images.shape[0]
        
        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        
        for _ in range(max_iter):
            adv_images = AdversarialAttacks._clamp_to_bounds(
                images + delta,
                clamp_min=clamp_min,
                clamp_max=clamp_max
            )
            outputs = model(adv_images)
            
            # C&W loss
            real = outputs.gather(1, labels.unsqueeze(1))
            other = outputs.clone()
            other.scatter_(1, labels.unsqueeze(1), -1e10)
            other_max = other.max(1)[0]
            
            # f-function
            f = torch.clamp(real.squeeze() - other_max + kappa, min=0)
            
            # Total loss
            l2_loss = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            loss = torch.mean(l2_loss + c * f)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return AdversarialAttacks._clamp_to_bounds(
            images + delta,
            clamp_min=clamp_min,
            clamp_max=clamp_max
        ).detach()


def generate_adversarial_dataset(model, dataloader, attack_type='fgsm', 
                                 epsilon=0.1, max_samples=1000):
    """
    Generate adversarial examples from a dataset.
    
    Args:
        model: Target model
        dataloader: DataLoader with clean examples
        attack_type: Type of attack ('fgsm', 'pgd', 'random', 'cw')
        epsilon: Attack strength
        max_samples: Maximum number of samples to generate
        
    Returns:
        tuple: (clean_images, adversarial_images, labels)
    """
    model.eval()
    
    clean_images_list = []
    adv_images_list = []
    labels_list = []
    
    attack_fn = {
        'fgsm': AdversarialAttacks.fgsm_attack,
        'pgd': AdversarialAttacks.pgd_attack,
        'random': AdversarialAttacks.random_noise_attack,
        'cw': AdversarialAttacks.carlini_wagner_l2
    }
    
    count = 0
    for images, labels in dataloader:
        if count >= max_samples:
            break
            
        # Generate adversarial examples
        if attack_type == 'random':
            adv_images = attack_fn[attack_type](images, epsilon)
        elif attack_type == 'cw':
            adv_images = attack_fn[attack_type](model, images, labels)
        else:
            adv_images = attack_fn[attack_type](model, images, labels, epsilon)
        
        clean_images_list.append(images)
        adv_images_list.append(adv_images)
        labels_list.append(labels)
        
        count += len(images)
    
    clean_images = torch.cat(clean_images_list, dim=0)
    adv_images = torch.cat(adv_images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    return clean_images, adv_images, all_labels
