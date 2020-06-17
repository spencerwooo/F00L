import cv2
import foolbox
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.distances import Linf
from matplotlib import rcParams

from utils import utils

CLASS_NAMES = [
  "tench",
  "English springer",
  "cassette player",
  "chain saw",
  "church",
  "French horn",
  "garbage truck",
  "gas pump",
  "golf ball",
  "parachute",
]

model = utils.load_trained_model(
  model_name="resnet",
  model_path="../models/200224_0901_resnet_imagenette.pth",
  class_num=len(CLASS_NAMES),
)
preprocessing = dict(
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
)

# resize image to size 213 * 213
transform = transforms.Compose(
  [transforms.Resize((213, 213)), transforms.ToTensor()]
)

# load dataset with validation images
dataset = torchvision.datasets.ImageFolder(
  root="../data/imagenette2-160/val", transform=transform
)

# 2. get first 100 images (all tenches)
dataset = torch.utils.data.Subset(dataset, [431])

# compose dataset into dataloader
# (don't shuffle, no need to shuffle, we're not training.)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
# get dataset size (length)
dataset_size = len(dataset)

# Use GPU if available
if torch.cuda.is_available():
  model = model.cuda()
fmodel = foolbox.models.PyTorchModel(
  model,
  bounds=(0, 1),
  num_classes=len(CLASS_NAMES),
  preprocessing=preprocessing,
)
rcParams["font.family"] = "monospace"


def img_to_np(img):
  """ Transpose image to viewable format to plot/visualize. """
  return np.transpose(img, (1, 2, 0))


attack = fa.GradientSignAttack(fmodel, distance=Linf)
for image, label in dataset_loader:
  plt.figure(figsize=(12, 5.4))
  plt.subplot(1, 3, 1)
  plt.imshow(img_to_np(image.squeeze()))

  prob = fmodel.forward(image.numpy())
  pred = np.argmax(prob, axis=-1)
  plt.title(
    "Original prediction:\n{}".format(CLASS_NAMES[pred[0]]), color="b",
  )

  adv = attack(image.numpy(), label.numpy(), epsilons=[4 / 255])
  prob = fmodel.forward(adv)
  pred = np.argmax(prob, axis=-1)
  plt.subplot(1, 3, 2)
  plt.imshow(img_to_np(adv.squeeze()))
  plt.title(
    "Adversary prediction:\n{}".format(CLASS_NAMES[pred[0]]), color="r",
  )

  # plt.subplot(1, 3, 3)
  # plt.imshow(img_to_np((adv - image.numpy()).squeeze()))
  # plt.title("Perturbation: $\\epsilon=4/255$")

  resized_adv = cv2.resize(
    np.moveaxis(adv.squeeze(), 0, 2),
    (0, 0),
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_LINEAR,
  )
  resized_adv = np.moveaxis(resized_adv, 2, 0)
  resized_adv = np.expand_dims(resized_adv, axis=1)

  prob = fmodel.forward(resized_adv)
  pred = np.argmax(prob, axis=-1)
  plt.subplot(1, 3, 3)
  plt.imshow(img_to_np(resized_adv.squeeze()))
  plt.title(
    "Resized adversary prediction:\n{}".format(CLASS_NAMES[pred[0]]), color="g",
  )
