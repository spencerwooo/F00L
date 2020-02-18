#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Applying an adversarial attack on a pretrained ResNet18 model `resnet_imagenette`:

- Pretrained on ImageNet
- Transfer training targeted on ImageNette
- Adversarial Attacks performed with Foolbox

Model weights: https://drive.google.com/open?id=1_YrCbnWwDMlFFRoldsv7PeoNf54yPeVW
ImageNette: https://github.com/fastai/imagenette
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
from tqdm.auto import tqdm

# pretrained model state_dict path
MODEL_PATH = './resnet_imagenette.pth'

# 10 classes
CLASS_NAMES = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
               'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

# instantiate model
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

# add final linear layer for feature extraction
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

# load pretrained weights
model.load_state_dict(torch.load(MODEL_PATH))
# set mode evaluation
model.eval()

print('Instantiated ConvNET model: ResNet18IMGNette.')

# %%
# use GPU if available
if torch.cuda.is_available():
  model = model.cuda()

# define preprocessing procedures (Foolbox)
preprocessing = dict(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225], axis=-3)

# define Foolbox wrapper (Foolbox)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1),
                                     num_classes=10,
                                     preprocessing=preprocessing)

# resize image to size 213 * 213
transform = transforms.Compose([
    transforms.Resize((213, 213)),
    transforms.ToTensor()
])

# training dataset path
dataset_path = '../data/imagenette2-160/val'

# load dataset with validation images
dataset = torchvision.datasets.ImageFolder(
    root=dataset_path, transform=transform)
# use first 1000 images (1000 images use approximately 30s on GPU)
dataset = torch.utils.data.Subset(dataset, range(0, 1000))
# compose dataset into dataloader
dataset_loader = torch.utils.data.DataLoader(
    dataset, shuffle=True, num_workers=0)
dataset_size = len(dataset)

print('Loaded data from: {} with a total of {} images.'.format(
    dataset_path, dataset_size))

# %%
# show sample images from ImageNette
LEN = 4
N_COL = 4
N_ROW = 1


def img_to_np(image):
  np_img = image.numpy()
  return np.transpose(np_img, (1, 2, 0))


plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
data_iter = iter(dataset_loader)
for i in range(LEN):
  sample_image, sample_label = data_iter.next()
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(img_to_np(sample_image.squeeze()))
  plt.title('GROUND TRUTH\n{}'.format(CLASS_NAMES[sample_label.squeeze()]))

# %%
pbar = tqdm(dataset_loader)
pbar.set_description('Validate predictions:')
pbar.set_postfix(acc='0.0%')

probs = []
acc = 0.0
i = 0
for image, label in pbar:
  # make a prediction
  prob = fmodel.forward(image.numpy())
  pred = np.argmax(prob)
  probs.append(prob)
  i += 1

  # calculate current accuracy
  acc += torch.sum(pred == label.data)
  current_acc = acc * 100 / i
  pbar.set_postfix(acc='{:.2f}%'.format(current_acc))
  # label_list.append(labels)

acc = acc * 100 / dataset_size
pbar.write('\nValidated with accuracy of: {:.2f}%'.format(acc))

# %%
# make a prediction
probs = fmodel.forward(images.numpy())

# plot predictions
plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
for i in range(LEN):
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(img_to_np(images[i]))
  plt.title('PREDICTION\n{}'.format(CLASS_NAMES[np.argmax(probs[i])]))

# %%
# perform an adversarial attack with FGSM
attack = foolbox.attacks.FGSM(fmodel)
# generate adversarial examples
adversarials = attack(images.numpy(), labels.numpy())
# make predictions on adversarial examples
adv_probs = fmodel.forward(adversarials)

# plot adversarial examples predictions
plt.figure(figsize=(N_COL * 2.5, N_ROW * 2))
for i in range(LEN):
  plt.subplot(N_ROW, N_COL, i + 1)
  plt.imshow(np.transpose(adversarials[i], (1, 2, 0)))
  plt.title('ADVERSARIAL\n{}'.format(CLASS_NAMES[np.argmax(adv_probs[i])]))
