#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Applying an adversarial attack on ResNet
with ImageNet weights

Training dataset with a subset of ImageNet: ImageNette
https://github.com/fastai/imagenette
"""

# %%
import foolbox
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# instantiate model
model = models.resnet18(pretrained=True).eval()

# %%
# use GPU if available
if torch.cuda.is_available():
  model = model.cuda()

preprocessing = dict(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225], axis=-3)

fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1),
                                     num_classes=1000,
                                     preprocessing=preprocessing)

# retrieve training dataset
transform = transforms.Compose([
    transforms.Resize((213, 213)),
    transforms.ToTensor()
])

# training dataset path
dataset_path = '../data/imagenette2-160/train'

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
  plt.title('GROUND TRUTH\n{}'.format(labels[i]))

# %%
probs = fmodel.forward(images.numpy())

# %%
attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images.numpy(), labels.numpy())
adv_probs = fmodel.forward(adversarials)

# %%
