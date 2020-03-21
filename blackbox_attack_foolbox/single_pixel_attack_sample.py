import time
import os

import foolbox
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# pretrained model state_dict path
MODEL_RESNET_PATH = '../resnet_foolbox/200224_0901_resnet_imagenette.pth'

# 10 classes
CLASS_NAMES = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump',
    'golf ball', 'parachute'
]

# instantiate model
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

# add final linear layer for feature extraction
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

# load pretrained weights
model.load_state_dict(torch.load(MODEL_RESNET_PATH))
# set mode evaluation
model.eval()

print('Instantiated ConvNET model: ResNet18ImageNette.')

# use GPU if available
if torch.cuda.is_available():
  model = model.cuda()

# define preprocessing procedures (Foolbox)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

# define Foolbox wrapper (Foolbox)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

# resize image to size 213 * 213
transform = transforms.Compose([transforms.Resize((213, 213)), transforms.ToTensor()])

class_start_indice = [indice * 200 for indice in range(0, 1)]
images_in_class_indice = np.array([[j for j in range(k, k + 1)] for k in class_start_indice]).flatten()
# training dataset path
dataset_path = '../data/imagenette2-160/val'

# load dataset with validation images
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# 1. get 10 images from 10 classes for a total of 100 images, or ...
dataset = torch.utils.data.Subset(dataset, images_in_class_indice)
# 2. get first 100 images (all tenches)
# dataset = torch.utils.data.Subset(dataset, range(0, 100))

# compose dataset into dataloader
# (don't shuffle, no need to shuffle, we're not training.)
dataset_loader = torch.utils.data.DataLoader(dataset)
# get dataset size (length)
dataset_size = len(dataset)

print('Loaded data from: {} with a total of {} images.'.format(dataset_path, dataset_size))

# Validate model's base prediction accuracy (about 97%)
print('\n[TASK 1/3] Validate orginal prediction:')
pbar = tqdm(dataset_loader)
pbar.set_description('Validate predictions')
pbar.set_postfix(acc='0.0%')

preds = []
acc = 0.0
i = 0
for image, label in pbar:
  # make a prediction
  prob = fmodel.forward(image.numpy())
  pred = np.argmax(prob)
  preds.append(pred)
  i += 1

  # calculate current accuracy
  acc += torch.sum(pred == label.data)
  current_acc = acc * 100 / i
  pbar.set_postfix(acc='{:.2f}%'.format(current_acc))

acc = acc * 100 / dataset_size
pbar.write('Validated with accuracy of: {:.2f}%'.format(acc))

# Perform an adversarial attack with FGSM
print('\n[TASK 2/3] Generate adversaries:')
tic = time.time()
attack = foolbox.attacks.SinglePixelAttack(model=fmodel,
                                           distance=foolbox.distances.Linf,
                                           criterion=foolbox.criteria.Misclassification())

pbar = tqdm(dataset_loader)
pbar.set_description('Generate adversaries')

# iterate through images to generate adversaries
adversaries = []
for image, label in pbar:
  adv = attack(image.numpy(), label.numpy())

  # if an attack fails under preferred criterions, `np.nan` is returned,
  #  in which case, we'll return the original image
  if np.isnan(adv).any():
    adv = image.numpy()
  adversaries.append(adv)

toc = time.time()
time_elapsed = toc - tic
pbar.write('Adversaries generated in: {:.2f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))

# make predictions on adversarial examples
print('\n[TASK 3/3] Validate adversaries:')
pbar = tqdm(dataset_loader)
pbar.set_description('Validate adversaries')
pbar.set_postfix(acc='0.00%')
i = 0
adv_acc = 0.0
adv_preds = []
adv_failed = []
for _, label in pbar:
  adv_prob = fmodel.forward(adversaries[i])
  adv_pred = np.argmax(adv_prob)
  adv_preds.append(adv_pred)

  # attack failed (adversarial == ground truth)
  if adv_pred == label.data:
    adv_failed.append(i)

  i += 1

  adv_acc += torch.sum(adv_pred == label.data)
  cur_adv_acc = adv_acc * 100 / i
  pbar.set_postfix(acc='{:.2f}%'.format(cur_adv_acc))

adv_acc = adv_acc * 100 / dataset_size
pbar.write('Model predicted adversaries with an accuracy of: {:.2f}%'.format(adv_acc))

# Send notifications
bitjs = '~/.net/BIT.js'
title = 'Attack complete'
msg = 'Time elapsed {:.2f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60)
cmd = 'python notify.py -b "{}" -t "{}" -m "{}"'.format(title, msg)
stream = os.popen(cmd)
output = stream.read()
print('\n' + output)
