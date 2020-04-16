#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adversarial attacks with Foolbox 3.0.

Implementing adversarial attacks with constrainted distances with
newest version of Foolbox: 3.0. (Still testing if viable.)

Todo:
  * L2DeepFoolAttack and L2CarliniWagnerAttack have not been
    successfully constrained. Still working on them.
  * Need to add condition to ignore fixed epsilon attacks.
  * Black box attacks like Boundary Attack have not been tested.

"""

import time

import foolbox.distances as fd
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import numpy as np
import torch
from foolbox import PyTorchModel
from numpy.linalg import norm
from tqdm import tqdm

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

NORM = "inf"  # "inf" or "2"
THRESHOLD = 5 # "inf": 4; "2": 5

BATCH_SIZE = 4
DATASET_IMAGE_NUM = 10
DATASET_PATH = "../data/imagenette2-160/val"

MODEL_RESNET_PATH = "../models/200224_0901_resnet_imagenette.pth"


def model_validate(fmodel, device, dataset_loader, dataset_size, adv=None):
  """ Benchmark model accuracy with either original images or advs. """

  pbar = tqdm(dataset_loader)
  pbar.set_description("Img" if adv is None else "Adv")
  acc = 0.0

  for i, (image, label) in enumerate(pbar):
    image = image.to(device)
    label = label.to(device)
    predictions = fmodel(image if adv is None else adv[i]).argmax(axis=-1)
    acc += (predictions == label).sum().item()

  print("{}: {}%".format("Img" if adv is None else "Adv", acc * 100 / dataset_size))
  # give the output to stdout a sec to show (same as below)
  time.sleep(0.5)


def plot_distances(dist, lp_norm="inf"):
  """ Plot each of the distances of the adversaries """

  indice = np.arange(0, len(dist), 1)

  plt.scatter(indice, dist)
  plt.ylabel("L_{} distance".format(lp_norm))
  plt.xlabel("Adversaries")
  plt.ylim(0, THRESHOLD * 2)
  plt.hlines(y=THRESHOLD, xmin=-10, xmax=len(dist) + 10, colors="r")

  plt.grid(axis="y")
  plt.title(
    "L_{}: min {:.4f}, mean {:.4f}, max {:.4f}".format(
      lp_norm, dist.min(), np.median(dist), dist.max()
    )
  )

  plt.show()


def main():
  """ Validate -> Attack -> Revalidate """

  model = utils.load_trained_model(
    model_name="resnet", model_path=MODEL_RESNET_PATH, class_num=len(CLASS_NAMES)
  )

  preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

  dataset_loader, dataset_size = utils.load_dataset(
    dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM
  )

  # use GPU if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

  # * 1/3: Validate original images.
  model_validate(fmodel, device, dataset_loader, dataset_size)

  # * 2/3: Perform adversarial attack.
  # attack = fa.LinfFastGradientAttack()
  attack = fa.LinfBasicIterativeAttack()
  # attack = fa.L2CarliniWagnerAttack()
  # attack = fa.L2DeepFoolAttack()
  eps = [THRESHOLD]

  pbar = tqdm(dataset_loader)
  pbar.set_description("Att")
  dist = []
  adversaries = []

  for image, label in pbar:
    advs, _, _ = attack(fmodel, image.to(device), label.to(device), epsilons=eps)

    for i, (single_adv, single_img) in enumerate(zip(advs[0], image.to(device))):
      perturb = (single_adv - single_img).cpu()
      _l_p = norm(perturb.flatten(), np.inf if NORM == "inf" else 2)
      _dist = fd.linf(single_img, single_adv)

      # Todo: add condition to ignore fixed epsilon attacks
      if _l_p > THRESHOLD:
        # replace adversaries with perturbations larger than threshold with original
        # images (for attacks with minimization approaches: cw, deepfool)
        # _l_p = 0.0
        advs[0][i] = single_img

      dist.append(_l_p)

    adversaries.append(advs[0])

  dist = np.asarray(dist)
  plot_distances(dist, lp_norm=NORM)
  # np.save("cw_dist_lim_1.npy", dist)

  print(
    "L_{}: min {:.4f}, mean {:.4f}, max {:.4f}".format(
      NORM, dist.min(), np.median(dist), dist.max()
    )
  )
  time.sleep(0.5)

  # * 3/3: Validate generated adversarial examples.
  model_validate(fmodel, device, dataset_loader, dataset_size, adv=adversaries)


if __name__ == "__main__":
  main()
