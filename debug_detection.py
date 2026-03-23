import torch
from models import SimpleCNN_MNIST
from detectors import AdversarialDetectionSystem
from utils import AdversarialAttacks
from torchvision import datasets, transforms

# load model
model = SimpleCNN_MNIST()
import os
if os.path.exists('trained_models/mnist_cnn.pth'):
    model.load_state_dict(torch.load('trained_models/mnist_cnn.pth', map_location='cpu'))
model.eval()

# load a single MNIST test sample
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image, label = dataset[0]
image = image.unsqueeze(0)
label = torch.tensor([label])

# generate adversarial image
epsilon = 0.1
adv = AdversarialAttacks.fgsm_attack(model, image, label, epsilon)

# compute perturbation score
from detectors.perturbation_detector import PerturbationDetector
pert = PerturbationDetector(threshold=0.1)
score = pert.calculate_perturbation(image, adv)
print('pert score', score)
print('threshold', pert.threshold)
print('is detected by pure perturbation', score > pert.threshold)

# run full detection system
detector = AdversarialDetectionSystem(model)
res = detector.detect(adv, baseline_input=image)
print('full detector result:')
print(res)
