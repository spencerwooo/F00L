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
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from foolbox.distances import Linf
from numpy.linalg import norm
from tqdm import tqdm

from utils import utils

now = datetime.now()

# * attack method:
# fgsm / bim / mim / deep_fool / cw | hop_skip_jump / single_pixel
ATTACK_METHOD = "fgsm"
# model to attack: resnet / vgg / mobilenet / inception
TARGET_MODEL = "resnet"
# TODO: perturbation threshold L∞ norm [4, 8, 16, 32]
THRESHOLD = 32 / 255

SAVE_ADVS = False
SCATTER_PLOT_DIST = True

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
ADV_SAVE_NAME = "{}_{:.3f}_adv.npy".format(now.strftime("%m%d_%H%M"), THRESHOLD)


def init_models(model_name):
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
  # print('Model "{}" initialized.'.format(model_name))
  return model


def attack_switcher(att, fmodel):
  switcher = {
    "fgsm": fa.GradientSignAttack(fmodel, distance=Linf),
    "bim": fa.LinfinityBasicIterativeAttack(fmodel, distance=Linf),
    "mim": fa.MomentumIterativeAttack(fmodel, distance=Linf),
    "deep_fool": fa.DeepFoolLinfinityAttack(fmodel, distance=Linf),
    "cw": fa.CarliniWagnerL2Attack(fmodel, distance=Linf),
    "hop_skip_jump": fa.HopSkipJumpAttack(fmodel),
    "single_pixel": fa.SinglePixelAttack(fmodel),
  }
  return switcher.get(att)


if __name__ == "__main__":
  model = init_models(TARGET_MODEL)
  preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

  dataset_loader, dataset_size = utils.load_dataset(
    dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM
  )

  # use GPU if available
  if torch.cuda.is_available():
    model = model.cuda()

  fmodel = foolbox.models.PyTorchModel(
    model, bounds=(0, 1), num_classes=len(CLASS_NAMES), preprocessing=preprocessing
  )

  # * 1/3: Validate model's base prediction accuracy (about 97%)
  # print('[TASK 1/3] Validate imgs:')
  utils.validate(fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE)

  # * 2/3: Perform an adversarial attack with blackbox attack
  # print('[TASK 2/3] Generate advs with "{}", threshold "{:.3f}":'.format(
  #     ATTACK_METHOD, THRESHOLD))
  attack = attack_switcher(ATTACK_METHOD, fmodel)

  tic = time.time()
  pbar = tqdm(dataset_loader)
  pbar.set_description("Generate adversaries")

  adversaries = []
  distances = []

  for image, label in pbar:
    if ATTACK_METHOD == "fgsm":
      adv = attack(image.numpy(), label.numpy(), unpack=False, epsilons=[THRESHOLD])

    elif ATTACK_METHOD in ["mim", "bim"]:
      adv = attack(
        image.numpy(),
        label.numpy(),
        unpack=False,
        binary_search=False,
        epsilon=THRESHOLD,
      )

    elif ATTACK_METHOD in ["deep_fool", "cw"]:
      adv = attack(image.numpy(), label.numpy(), unpack=False)

    else:
      raise NotImplementedError

    adv_batch = []
    for i, (single_adv, single_img) in enumerate(zip(adv, image.numpy())):
      # TODO: perturbation distance under L2 norm
      a = single_adv.perturbed
      dist = single_adv.distance.value
      single_adv = a

      if single_adv is None:
        #! if an attack failed, replace adv with original image
        single_adv = single_img
      else:
        #! limit generated adversary perturbation size
        perturb = single_adv - single_img
        # _l2 = norm(perturb.flatten(), 2)
        _linf = norm(perturb.flatten(), np.inf)
        distances.append(_linf)

      adv_batch.append(single_adv)
    adversaries.append(adv_batch)

  adversaries = np.array(adversaries)

  # total attack time
  toc = time.time()
  time_elapsed = toc - tic
  pbar.write(
    "Adversaries generated in: {:.2f}m {:.2f}s".format(
      time_elapsed // 60, time_elapsed % 60
    )
  )

  #! evaluate mean distance
  distances = np.asarray(distances)
  # np.save('dist_{}.npy'.format(ATTACK_METHOD), distances)
  pbar.write(
    "Distance: min {:.5f}, mean {:.5f}, max {:.5f}".format(
      distances.min(), np.median(distances), distances.max()
    )
  )

  #! whether or not to plot the distances
  if SCATTER_PLOT_DIST:
    indice = np.arange(0, len(distances), 1)
    plt.scatter(indice, distances)
    plt.ylabel("L∞ distance")
    # plt.ylim(0, THRESHOLD * 2)
    plt.xlabel("Adversaries")
    plt.grid(axis="y")
    plt.show()

  # save generated adversaries
  if SAVE_ADVS:
    if not os.path.exists(ADV_SAVE_PATH):
      os.makedirs(ADV_SAVE_PATH)
    np.save(os.path.join(ADV_SAVE_PATH, ADV_SAVE_NAME), adversaries)

  # * 3/3: Validate model's adversary predictions
  # print('[TASK 3/3] Validate advs:')
  utils.validate(
    fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE, advs=adversaries
  )

  # notify
  # utils.notify(time_elapsed)
