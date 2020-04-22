"""
Perform adversarial attacks on CNNs

* Generated adversaries are saved inside:

  'advs/{TARGET_MODEL}/{ATTACK_METHOD}/{FILE_NAME}.npy'

* e.g.: 'advs/resnet/fgsm/0405_1021_0.02_adv.npy'
"""

import os
import time
from datetime import datetime

import foolbox
import foolbox.attacks as fa
import matplotlib.pyplot as plt
import numpy as np
import torch
from foolbox.criteria import TargetClass
from foolbox.distances import Linf
from matplotlib import rcParams
from numpy.linalg import norm
from tqdm import tqdm

from utils import utils

NOW = datetime.now()

# Methods: fgsm / bim / mim / df / cw | hsj / ga
ATTACK_METHOD = "hsj"
# Models: resnet / vgg / mobilenet / inception
TARGET_MODEL = "resnet"
# Perturbation threshold: L∞ - 8/255, L2 - 5 (GenAttack: L∞ - 0.5)
THRESHOLD = 0.5

SAVE_DIST = False
SAVE_ADVS = True
SCATTER_PLOT_DIST = False

MODEL_RESNET_PATH = "../models/200224_0901_resnet_imagenette.pth"
MODEL_VGG_PATH = "../models/200226_0225_vgg11_imagenette.pth"
MODEL_MOBILENET_PATH = "../models/200226_0150_mobilenet_v2_imagenette.pth"
MODEL_INCEPTION_PATH = "../models/200228_1003_inception_v3_imagenette.pth"

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

BATCH_SIZE = 4
# testing: 1 x 1, normal: 10 x 10
DATASET_IMAGE_NUM = 10
DATASET_PATH = "../data/imagenette2-160/val"
ADV_SAVE_PATH = os.path.join("advs", TARGET_MODEL, ATTACK_METHOD)
ADV_SAVE_NAME = "{}_{:.3f}_adv.npy".format(NOW.strftime("%m%d_%H%M"), THRESHOLD)


def init_models(model_name):
  """ Initialize pretrained CNN models """

  model_path = {
    "resnet": MODEL_RESNET_PATH,
    "vgg": MODEL_VGG_PATH,
    "mobilenet": MODEL_MOBILENET_PATH,
    "inception": MODEL_INCEPTION_PATH,
  }

  model = utils.load_trained_model(
    model_name=model_name,
    model_path=model_path.get(model_name),
    class_num=len(CLASS_NAMES),
  )
  return model


def attack_switcher(att, fmodel):
  """ Initialize different attacks. """

  switcher = {
    "fgsm": fa.GradientSignAttack(fmodel, distance=Linf),
    "bim": fa.LinfinityBasicIterativeAttack(fmodel, distance=Linf),
    "mim": fa.MomentumIterativeAttack(fmodel, distance=Linf),
    "df": fa.DeepFoolL2Attack(fmodel),
    "cw": fa.CarliniWagnerL2Attack(fmodel),
    "hsj": fa.HopSkipJumpAttack(fmodel, distance=Linf),
    "ga": fa.GenAttack(fmodel, criterion=TargetClass(9), distance=Linf),
  }

  return switcher.get(att)


def attack_params(att, image, label):
  """ Inject attack parameters into attack() function """

  params = {
    "fgsm": {"epsilons": [THRESHOLD]},
    "mim": {"binary_search": False, "epsilon": THRESHOLD},
    "bim": {"binary_search": False, "epsilon": THRESHOLD},
    "df": {},
    "cw": {},
    "hsj": {"batch_size": BATCH_SIZE},
    "ga": {"binary_search": False, "epsilon": THRESHOLD},
  }

  for key in params:
    params[key]["inputs"] = image.numpy()
    params[key]["labels"] = label.numpy()

  return params.get(att)


def plot_distances(distances):
  """ Plot distances between adversaries and originals. """

  rcParams["font.family"] = "monospace"

  indice = np.arange(0, len(distances), 1)
  plt.scatter(indice, distances)
  plt.hlines(y=THRESHOLD, xmin=0, xmax=len(distances), colors="r")

  plt.ylabel("Distance")
  plt.ylim(0, THRESHOLD * 2)

  plt.xlabel("Adversaries")

  plt.title("Attack: {} - Threshold: {:.5f}".format(ATTACK_METHOD, THRESHOLD))
  plt.grid(axis="y")
  plt.show()


def main():
  """ Validate -> Attack -> Revalidate """

  model = init_models(TARGET_MODEL)
  preprocessing = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
  )

  dataset_loader, dataset_size = utils.load_dataset(
    dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM
  )

  # Use GPU if available
  if torch.cuda.is_available():
    model = model.cuda()

  fmodel = foolbox.models.PyTorchModel(
    model,
    bounds=(0, 1),
    num_classes=len(CLASS_NAMES),
    preprocessing=preprocessing,
  )

  # * 1/3: Validate model's base prediction accuracy (about 97%)
  utils.validate(fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE)

  # * 2/3: Perform an adversarial attack with blackbox attack
  attack = attack_switcher(ATTACK_METHOD, fmodel)

  tic = time.time()
  pbar = tqdm(dataset_loader)
  pbar.set_description("ATT")

  adversaries = []
  distances = []

  for image, label in pbar:
    adv = attack(**attack_params(ATTACK_METHOD, image, label))

    adv_batch = []
    for single_adv, single_img in zip(adv, image.numpy()):

      # If an attack failed, replace adv with original image
      if np.isnan(single_adv).any():
        single_adv = single_img

      perturb = single_adv - single_img

      # Only DeepFool and CW attacks are evaluated with L2 norm
      _lp = norm(
        perturb.flatten(), 2 if ATTACK_METHOD in ["cw", "df"] else np.inf
      )

      # For attacks with minimization approaches (deep fool, cw),
      # if distance larger than threshold, we consider attack failed
      if _lp > THRESHOLD and ATTACK_METHOD in ["df", "cw", "hsj"]:
        _lp = 0.0
        single_adv = single_img

      if np.isnan(_lp):
        _lp = 0.0

      distances.append(_lp)

      adv_batch.append(single_adv)
    adversaries.append(adv_batch)

  adversaries = np.array(adversaries)

  # Total attack time
  toc = time.time()
  time_elapsed = toc - tic
  print(
    "Adversaries generated in: {:.2f}m {:.2f}s".format(
      time_elapsed // 60, time_elapsed % 60
    )
  )

  #! evaluate mean distance
  distances = np.asarray(distances)
  if SAVE_DIST:
    np.save("dist_{}.npy".format(ATTACK_METHOD), distances)
  print(
    "Distance: min {:.5f}, mean {:.5f}, max {:.5f}".format(
      distances.min(), np.median(distances), distances.max()
    )
  )
  time.sleep(0.5)

  # Whether or not to plot the distances
  if SCATTER_PLOT_DIST:
    plot_distances(distances)

  # Save generated adversaries
  if SAVE_ADVS:
    if not os.path.exists(ADV_SAVE_PATH):
      os.makedirs(ADV_SAVE_PATH)
    np.save(os.path.join(ADV_SAVE_PATH, ADV_SAVE_NAME), adversaries)

  # * 3/3: Validate model's adversary predictions
  utils.validate(
    fmodel,
    dataset_loader,
    dataset_size,
    batch_size=BATCH_SIZE,
    advs=adversaries,
  )

  # Notify
  # utils.notify(time_elapsed)


if __name__ == "__main__":
  main()
