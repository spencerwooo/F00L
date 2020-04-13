# -*- coding: utf-8 -*-
"""Adversarial attacks with Foolbox 3.0.

Implementing adversarial attacks with constrainted distances with
newest version of Foolbox: 3.0. (Still testing if viable.)

Todo:
  * L2DeepFoolAttack and L2CarliniWagnerAttack have not been
    successfully constrained. Still working on them.
  * Black box attacks like Boundary Attack have not been tested.

"""

import foolbox.attacks as fa
import matplotlib.pyplot as plt
import numpy as np
import torch
from foolbox import PyTorchModel
from numpy.linalg import norm
from tqdm import tqdm

from utils import utils

CLASS_NAMES = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church',
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
]
BATCH_SIZE = 4
DATASET_IMAGE_NUM = 10
DATASET_PATH = '../data/imagenette2-160/val'
MODEL_RESNET_PATH = '../models/200224_0901_resnet_imagenette.pth'


def model_validate(fmodel, device, dataset_loader, dataset_size, adv=None):
  """ Benchmark model accuracy with either original images or advs. """

  pbar = tqdm(dataset_loader)
  acc = 0.0

  for i, (image, label) in enumerate(pbar):
    image = image.to(device)
    label = label.to(device)
    predictions = fmodel(image if adv is None else adv[i]).argmax(axis=-1)
    acc += (predictions == label).sum().item()

  print('{}: {}%'.format('Imgs' if adv is None else 'Advs',
                         acc * 100 / dataset_size))


def plot_distances(dist, lp_norm='inf'):
  """ Plot each of the distances of the adversaries """

  indice = np.arange(0, len(dist), 1)
  plt.scatter(indice, dist)
  plt.ylabel('L{} distance'.format(lp_norm))
  plt.xlabel('Adversaries')
  plt.grid(axis='y')
  plt.show()


def main():
  """ Validate -> Attack -> Revalidate """

  model = utils.load_trained_model(model_name='resnet',
                                   model_path=MODEL_RESNET_PATH,
                                   class_num=len(CLASS_NAMES))

  preprocessing = dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       axis=-3)

  dataset_loader, dataset_size = utils.load_dataset(
      dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM)

  # use GPU if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

  #* 1/3: Validate original images.
  model_validate(fmodel, device, dataset_loader, dataset_size)

  #* 2/3: Perform adversarial attack.
  attack = fa.LinfBasicIterativeAttack()
  # attack = fa.L2CarliniWagnerAttack()
  eps = [4 / 255]

  pbar = tqdm(dataset_loader)
  dist = []
  adversaries = []

  for image, label in pbar:
    advs, _, _ = attack(fmodel,
                        image.to(device),
                        label.to(device),
                        epsilons=eps)
    adversaries.append(advs[0])

    for single_adv, single_img in zip(advs[0], image.to(device)):
      perturb = (single_adv - single_img).cpu()
      _linf = norm(perturb.flatten(), np.inf)
      # _l2 = norm(perturb.flatten(), 2)
      dist.append(_linf)

  lp_norm = 'inf'
  dist = np.asarray(dist)
  plot_distances(dist, lp_norm=lp_norm)

  print('L_{}: min {:.3f}, mean {:.3f}, max {:.3f}'.format(
      lp_norm, dist.min(), np.median(dist), dist.max()))

  #* 3/3: Validate generated adversarial examples.
  model_validate(fmodel, device, dataset_loader, dataset_size, adv=adversaries)


if __name__ == "__main__":
  main()
