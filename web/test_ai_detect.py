"""Test AI detection with generated test images."""
import requests
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

tests = [
    ("test_images/real_dog.png",      "Real Photo - Dog (should be Real/Likely Real)"),
    ("test_images/real_coffee.png",   "Real Photo - Coffee (should be Real/Likely Real)"),
    ("test_images/ai_cat.png",        "AI Art - Cyberpunk Cat (should be AI-Generated)"),
    ("test_images/ai_landscape.png",  "AI Art - Fantasy Landscape (should be AI-Generated)"),
]

for path, label in tests:
    files = {'image': open(path, 'rb')}
    r = requests.post('http://localhost:5000/api/detect_ai', files=files)
    d = r.json()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Verdict:  {d.get('verdict','?')}")
    print(f"  Score:    {d.get('ensemble_score', 0) * 100:.1f}%")
    print("  Detectors:")
    for k, v in d.get('detectors', {}).items():
        print(f"    {v['name']:30s} -> {v['score']:.4f}")
    print()
