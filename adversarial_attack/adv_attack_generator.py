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
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from foolbox.distances import Linf
from tqdm import tqdm

from utils import utils

# get current time
now = datetime.now()

#* attack method:
# white box: fgsm / deep_fool / jsma / cw / mi_fgsm
# black box: hop_skip_jump / single_pixel
ATTACK_METHOD = 'fgsm'
# model to attack: resnet / vgg / mobilenet / inception
TARGET_MODEL = 'resnet'
# perturbation threshold [4, 8, 16, 32]
# TODO: threshold evaluation, Linfinity?
PERTURB_THRESHOLD = 32 / 255

# pretrained model state_dict path
MODEL_RESNET_PATH = '../models/200224_0901_resnet_imagenette.pth'
MODEL_VGG_PATH = '../models/200224_0839_vgg_imagenette.pth'
MODEL_MOBILENET_PATH = '../models/200226_0150_mobilenet_v2_imagenette.pth'
MODEL_INCEPTION_PATH = '../models/200228_1003_inception_v3_imagenette.pth'

# 10 classes
CLASS_NAMES = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church',
    'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
]

# size of each batch
BATCH_SIZE = 4
# testing: 1 x 1, normal: 10 x 10
DATASET_IMAGE_NUM = 10
# training dataset path
DATASET_PATH = '../data/imagenette2-160/val'
# adv save path, name
ADV_SAVE_PATH = os.path.join('advs', TARGET_MODEL, ATTACK_METHOD)
ADV_SAVE_NAME = '{}_{:.2f}_adv.npy'.format(now.strftime('%m%d_%H%M'),
                                           PERTURB_THRESHOLD)


def init_models(model_name):
  model_path = {
      'resnet': MODEL_RESNET_PATH,
      'vgg': MODEL_VGG_PATH,
      'mobilenet': MODEL_MOBILENET_PATH,
      'inception': MODEL_INCEPTION_PATH
  }

  # instantiate resnet model
  model = utils.load_trained_model(model_name=model_name,
                                   model_path=model_path.get(model_name),
                                   class_num=len(CLASS_NAMES))
  print('Model "{}" initialized.'.format(model_name))
  return model


def attack_switcher(att, fmodel):
  switcher = {
      'fgsm': foolbox.attacks.GradientSignAttack(fmodel, distance=Linf),
      'deep_fool': foolbox.attacks.DeepFoolAttack(fmodel, distance=Linf),
      'jsma': foolbox.attacks.SaliencyMapAttack(fmodel, distance=Linf),
      'cw': foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=Linf),
      'mi_fgsm': foolbox.attacks.MomentumIterativeAttack(fmodel, distance=Linf),
      'hop_skip_jump': foolbox.attacks.HopSkipJumpAttack(fmodel, distance=Linf),
      'single_pixel': foolbox.attacks.SinglePixelAttack(fmodel, distance=Linf)
  }
  return switcher.get(att)


def main():
  # load models
  model = init_models(TARGET_MODEL)

  # define preprocessing procedures (foolbox)
  preprocessing = dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       axis=-3)

  # load dataset
  dataset_loader, dataset_size = utils.load_dataset(
      dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM)

  # use GPU if available
  if torch.cuda.is_available():
    model = model.cuda()

  # define foolbox wrapper
  fmodel = foolbox.models.PyTorchModel(model,
                                       bounds=(0, 1),
                                       num_classes=len(CLASS_NAMES),
                                       preprocessing=preprocessing)

  #* 1/3: Validate model's base prediction accuracy (about 97%)
  print('[TASK 1/3] Validate original prediction:')
  utils.validate(fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE)

  #* 2/3: Perform an adversarial attack with blackbox attack
  print(
      '[TASK 2/3] Generate adversaries with "{}" on threshold "{:.4f}":'.format(
          ATTACK_METHOD, PERTURB_THRESHOLD))
  attack = attack_switcher(ATTACK_METHOD, fmodel)

  tic = time.time()
  pbar = tqdm(dataset_loader)
  pbar.set_description('Generate adversaries')

  # iterate through images to generate adversaries
  adversaries = []
  distances = []
  for image, label in pbar:
    adv = attack(image.numpy(), label.numpy(), unpack=False)

    adv_batch = []
    for i, (single_adv, single_img) in enumerate(zip(adv, image.numpy())):
      #TODO: perturbation distance under L∞
      a = single_adv.perturbed
      dist = single_adv.distance.value

      #! limit generated adversary perturbation size
      if (dist > PERTURB_THRESHOLD or a is None):
        a = single_img
        # print('Perturbation too large: {:.2f}'.format(dist))
      else:
        distances.append(dist)

      adv_batch.append(a)

    # return adversary batch array
    adversaries.append(adv_batch)

  # construct np array from generated adversaries
  adversaries = np.array(adversaries)

  # total attack time
  toc = time.time()
  time_elapsed = toc - tic
  pbar.write('Adversaries generated in: {:.2f}m {:.2f}s'.format(
      time_elapsed // 60, time_elapsed % 60))

  # evaluate mean distance
  distances = np.asarray(distances)
  pbar.write('Distance: min {:.5f}, mean {:.5f}, max {:.5f}'.format(
      distances.min(), np.median(distances), distances.max()))

  # save generated adversaries
  if not os.path.exists(ADV_SAVE_PATH):
    os.makedirs(ADV_SAVE_PATH)
  np.save(os.path.join(ADV_SAVE_PATH, ADV_SAVE_NAME), adversaries)

  #* 3/3: Validate model's adversary predictions
  print('[TASK 3/3] Validate adversaries:')
  utils.validate(fmodel,
                 dataset_loader,
                 dataset_size,
                 batch_size=BATCH_SIZE,
                 advs=adversaries)

  # notify
  # utils.notify(time_elapsed)


if __name__ == "__main__":
  main()
