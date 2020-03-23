"""
Perform HopSkipJumpAttack on CNNs

* Generated adversaries are saved inside: 'advs/{ATTACK_METHOD}.npy'
* e.g.: 'advs/hop_skip_jump_attack_adv.npy'
"""

import os
import time

import foolbox
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import utils

# attack method: hop_skip_jump_attack / single_pixel_attack
ATTACK_METHOD = 'single_pixel_attack'
# model to attack: resnet / vgg / mobilenet / inception
TARGET_MODEL = 'resnet'

# pretrained model state_dict path
MODEL_RESNET_PATH = '../resnet_foolbox/200224_0901_resnet_imagenette.pth'
MODEL_VGG_PATH = '../vgg_foolbox/200224_0839_vgg_imagenette.pth'
MODEL_MOBILENET_PATH = '../mobilenet_foolbox/200226_0150_mobilenet_v2_imagenette.pth'
MODEL_INCEPTION_PATH = '../inception_foolbox/200228_1003_inception_v3_imagenette.pth'

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
# adv save path
ADV_SAVE_PATH = 'advs/{}_adv.npy'.format(ATTACK_METHOD)


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
  print('Instantiated ResNet18 with ImageNette trained weights.')
  return model


def attack_switcher(att, fmodel):
  switcher = {
      'hop_skip_jump_attack':
      foolbox.attacks.HopSkipJumpAttack(
          model=fmodel,
          distance=foolbox.distances.Linf,
          criterion=foolbox.criteria.Misclassification()),
      'single_pixel_attack':
      foolbox.attacks.SinglePixelAttack(
          model=fmodel,
          distance=foolbox.distances.Linf,
          criterion=foolbox.criteria.Misclassification())
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
  print('[TASK 2/3] Generate adversaries:')
  attack = attack_switcher(ATTACK_METHOD, fmodel)

  tic = time.time()
  pbar = tqdm(dataset_loader)
  pbar.set_description('Generate adversaries')

  # iterate through images to generate adversaries
  adversaries = []
  for image, label in pbar:
    if ATTACK_METHOD == 'hop_skip_jump_attack':
      adv = attack(image.numpy(), label.numpy())
    elif ATTACK_METHOD == 'single_pixel_attack':
      adv = attack(image.numpy(), label.numpy(), max_pixels=1000)

    # if an attack fails under preferred criterions, `np.nan` is returned,
    #  in which case, we'll return the original image
    for i, (single_adv, single_img) in enumerate(zip(adv, image.numpy())):
      if np.isnan(single_adv).any():
        adv[i] = single_img
    adversaries.append(adv)

  toc = time.time()
  time_elapsed = toc - tic
  pbar.write('Adversaries generated in: {:.2f}m {:.2f}s'.format(
      time_elapsed // 60, time_elapsed % 60))

  # save generated adversaries
  np.save(ADV_SAVE_PATH, adversaries)

  #* 3/3: Validate model's adversary predictions
  print('[TASK 3/3] Validate adversaries:')
  utils.validate(fmodel,
                 dataset_loader,
                 dataset_size,
                 batch_size=BATCH_SIZE,
                 advs=adversaries)

  # notify
  utils.notify(time_elapsed)


if __name__ == "__main__":
  main()
