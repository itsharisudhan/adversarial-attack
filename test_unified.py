"""
Comprehensive Real vs AI Detection Test

Tests the unified detector on:
  1. Real images (from ImageNette + existing test set)
  2. AI-generated images (Stable Diffusion / generative AI)
  3. Adversarially perturbed images (FGSM attack on real images)
  4. Duplicate submission detection

Prints a formatted results table with verdicts and scores.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from detectors.unified_detector import UnifiedDetector

# Also run forensic analysis from web/app.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))
from app import analyze_image


def separator():
    print("=" * 90)


def print_result(label, filename, result, forensic=None):
    verdict = result["verdict"]
    conf = result["confidence"]
    adv = result["adversarial_score"]
    genai = result.get("genai_score", 0.0)
    dup = result.get("is_duplicate", False)
    lid = result.get("details", {}).get("lid_score", "-")
    maha = result.get("details", {}).get("mahalanobis_distance", "-")

    # Status indicator
    verdict_display = verdict
    if verdict == "CLEAN":
        verdict_display = "CLEAN            [OK]"
    elif verdict == "AI_GENERATED":
        verdict_display = "AI_GENERATED     [DETECTED]"
    elif verdict == "ADVERSARIAL_ATTACK":
        verdict_display = "ADVERSARIAL      [DETECTED]"
    elif verdict == "TAMPERED":
        verdict_display = "TAMPERED         [DETECTED]"
    elif verdict == "DUPLICATE_FRAUD":
        verdict_display = "DUPLICATE_FRAUD  [DETECTED]"
    elif verdict == "HYBRID_THREAT":
        verdict_display = "HYBRID_THREAT    [DETECTED]"

    print(f"  [{label}] {filename}")
    print(f"    Verdict:     {verdict_display}")
    print(f"    Confidence:  {conf:.4f}")
    print(f"    Adv Score:   {adv:.4f}    GenAI Score: {genai:.4f}")
    print(f"    LID:         {lid}       Maha: {maha}")
    if forensic:
        det = forensic.get("detectors", {})
        fft = det.get("fft", {}).get("score", 0)
        ela = det.get("ela", {}).get("score", 0)
        stat = det.get("statistics", {}).get("score", 0)
        tex = det.get("texture", {}).get("score", 0)
        print(f"    Forensic:    FFT={fft:.3f}  ELA={ela:.3f}  Stats={stat:.3f}  Texture={tex:.3f}")
    print()


def main():
    separator()
    print("  UNIFIED FRAUD DETECTION SYSTEM - Real-Time Accuracy Test")
    separator()
    print()

    # Initialize detector
    detector = UnifiedDetector()
    print(f"  Adversarial Head: {'READY' if detector.adversarial_head_ready else 'DISABLED'}")
    print()

    real_dir = os.path.join("test_data", "real")
    ai_dir = os.path.join("test_data", "ai_generated")

    real_images = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ai_images = sorted([f for f in os.listdir(ai_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # --- Test 1: Real Images ---
    separator()
    print("  TEST 1: REAL IMAGES (should be CLEAN)")
    separator()
    print()

    real_correct = 0
    real_total = 0
    for fname in real_images:
        path = os.path.join(real_dir, fname)
        img = Image.open(path).convert("RGB")
        with open(path, "rb") as f:
            img_bytes = f.read()

        forensic = analyze_image(img, image_bytes=img_bytes)
        result = detector.detect(img, image_bytes=img_bytes, forensic_result=forensic)

        print_result("REAL", fname, result, forensic)
        real_total += 1
        if result["verdict"] == "CLEAN":
            real_correct += 1

    # --- Test 2: AI-Generated Images ---
    separator()
    print("  TEST 2: AI-GENERATED IMAGES (should be AI_GENERATED or TAMPERED)")
    separator()
    print()

    ai_correct = 0
    ai_total = 0
    for fname in ai_images:
        path = os.path.join(ai_dir, fname)
        img = Image.open(path).convert("RGB")
        with open(path, "rb") as f:
            img_bytes = f.read()

        forensic = analyze_image(img, image_bytes=img_bytes)
        result = detector.detect(img, image_bytes=img_bytes, forensic_result=forensic)

        print_result("AI", fname, result, forensic)
        ai_total += 1
        # Any non-CLEAN verdict = fraud detected (image correctly flagged)
        if result["verdict"] != "CLEAN":
            ai_correct += 1

    # --- Test 3: Adversarial Attack (add noise to real image) ---
    separator()
    print("  TEST 3: ADVERSARIAL PERTURBATION (FGSM-like noise on real images)")
    separator()
    print()

    adv_correct = 0
    adv_total = 0

    # Use real gradient-based FGSM via EfficientNet backbone
    import torch
    from torchvision import models, transforms
    from detectors.feature_extractor import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

    # Load full EfficientNet-B0 (with classifier head for gradient computation)
    adv_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    adv_model.eval()

    adv_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    epsilon = 0.10  # FGSM perturbation strength (stronger = more detectable)

    for fname in real_images[:5]:
        path = os.path.join(real_dir, fname)
        img = Image.open(path).convert("RGB")

        # Prepare tensor with gradient tracking
        x = adv_transform(img).unsqueeze(0)
        x.requires_grad = True

        # Forward pass -> get predicted class
        output = adv_model(x)
        pred_class = output.argmax(dim=1)

        # Compute loss and gradient
        loss = torch.nn.functional.cross_entropy(output, pred_class)
        adv_model.zero_grad()
        loss.backward()

        # FGSM: perturb in sign of gradient direction
        grad_sign = x.grad.data.sign()
        x_adv = x + epsilon * grad_sign
        x_adv = torch.clamp(x_adv, -3.0, 3.0)  # keep in reasonable range

        # Convert back to PIL for detection
        mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_adv_t = x_adv.squeeze(0).detach() * std_t + mean_t
        img_adv_t = torch.clamp(img_adv_t, 0, 1)
        adv_np = (img_adv_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        adv_img = Image.fromarray(adv_np)

        forensic = analyze_image(adv_img)
        result = detector.detect(adv_img, forensic_result=forensic)

        print_result("ADV", f"{fname} + FGSM(e={epsilon})", result, forensic)
        adv_total += 1
        # Any non-CLEAN verdict = attack was detected
        if result["verdict"] != "CLEAN":
            adv_correct += 1

    # --- Test 4: Duplicate Detection ---
    separator()
    print("  TEST 4: DUPLICATE SUBMISSION (re-upload same image)")
    separator()
    print()

    dup_correct = 0
    dup_total = 0
    # Pick first real image and re-upload it
    if real_images:
        fname = real_images[0]
        path = os.path.join(real_dir, fname)
        img = Image.open(path).convert("RGB")
        with open(path, "rb") as f:
            img_bytes = f.read()

        result = detector.detect(img, image_bytes=img_bytes)
        print_result("DUP", f"{fname} (second submission)", result)
        dup_total += 1
        if result["verdict"] == "DUPLICATE_FRAUD":
            dup_correct += 1

    # --- Summary ---
    separator()
    print("  ACCURACY SUMMARY")
    separator()
    print()
    print(f"  Real -> CLEAN:          {real_correct}/{real_total}  ({100*real_correct/max(real_total,1):.0f}%)")
    print(f"  AI -> AI_GENERATED:     {ai_correct}/{ai_total}  ({100*ai_correct/max(ai_total,1):.0f}%)")
    print(f"  Adversarial -> DETECTED:{adv_correct}/{adv_total}  ({100*adv_correct/max(adv_total,1):.0f}%)")
    print(f"  Duplicate -> DETECTED:  {dup_correct}/{dup_total}  ({100*dup_correct/max(dup_total,1):.0f}%)")
    print()

    total_correct = real_correct + ai_correct + adv_correct + dup_correct
    total_tests = real_total + ai_total + adv_total + dup_total
    print(f"  OVERALL ACCURACY:      {total_correct}/{total_tests}  ({100*total_correct/max(total_tests,1):.0f}%)")
    print()
    separator()


if __name__ == "__main__":
    main()
