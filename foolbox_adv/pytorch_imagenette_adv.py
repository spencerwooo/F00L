#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Applying an adversarial attack on ResNet with ImageNette weights
"""

# %%
import foolbox
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


CLASS_NAMES = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
               'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
MODEL_PATH = './resnet_imagenette.pth'

# instantiate model
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

# freeze all layers except the final layer (dense layer)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

# load pretrained weights
model.load_state_dict(torch.load(MODEL_PATH))

# %%
# use GPU if available
if torch.cuda.is_available():
  model = model.cuda()

preprocessing = dict(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225], axis=-3)

fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1),
                                     num_classes=10,
                                     preprocessing=preprocessing)

# retrieve training dataset
transform = transforms.Compose([
    transforms.Resize((213, 213)),
    transforms.ToTensor()
])

# training dataset path
dataset_path = '../data/imagenette2-160/val'

dataset = torchvision.datasets.ImageFolder(
    root=dataset_path, transform=transform)
dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0)

print('Loaded data from: {}'.format(dataset_path))

# %%
# show sample images from ImageNette
LEN = 4
N_COL = 4
N_ROW = 1

data_iter = iter(dataset_loader)
images, labels = data_iter.next()


def img_to_np(image):
  np_img = image.numpy()
  return np.transpose(np_img, (1, 2, 0))


plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
for i in range(LEN):
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(img_to_np(images[i]))
  plt.title('GROUND TRUTH\n{}'.format(CLASS_NAMES[labels[i]]))

# %%
probs = fmodel.forward(images.numpy())

plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
for i in range(LEN):
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(img_to_np(images[i]))
  plt.title('PREDICTION\n{}'.format(CLASS_NAMES[np.argmax(probs[i])]))

# %%
attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images.numpy(), labels.numpy())
adv_probs = fmodel.forward(adversarials)

plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
for i in range(LEN):
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(np.transpose(adversarials[i], (1, 2, 0)))
  plt.title('ADVERSARIAL\n{}'.format(CLASS_NAMES[np.argmax(adv_probs[i])]))
# %%
