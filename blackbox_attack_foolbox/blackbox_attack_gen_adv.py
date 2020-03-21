"""
Perform HopSkipJumpAttack on CNNs

* Generated adversaries are saved inside 'advs/hop_skip_jump_attack_adv.npy'
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

# pretrained model state_dict path
MODEL_RESNET_PATH = '../resnet_foolbox/200224_0901_resnet_imagenette.pth'
MODEL_VGG_PATH = '../vgg_foolbox/200224_0839_vgg_imagenette.pth'

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
# attack method
ATTACK_METHOD = 'hop_skip_jump_attack'
# adv save path
ADV_SAVE_PATH = 'advs/{}_adv.npy'.format(ATTACK_METHOD)


def init_models():
  # instantiate resnet model
  model = utils.load_trained_model(model_name='resnet',
                                   model_path=MODEL_RESNET_PATH,
                                   class_num=len(CLASS_NAMES))
  print('Instantiated ResNet18 with ImageNette trained weights.')

  # # instantiate vgg model
  # model = utils.load_trained_model(model_name='vgg',
  #                                      model_path=MODEL_VGG_PATH,
  #                                      class_num=len(CLASS_NAMES))
  # print('Instantiated VGG11, with ImageNette trained weights.')
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
  model = init_models()

  # define preprocessing procedures (Foolbox)
  preprocessing = dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       axis=-3)

  # load dataset
  dataset_loader, dataset_size = utils.load_dataset(
      dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM)

  # perform attack for loaded model

  # use GPU if available
  if torch.cuda.is_available():
    model = model.cuda()

  # define foolbox wrapper
  fmodel = foolbox.models.PyTorchModel(model,
                                       bounds=(0, 1),
                                       num_classes=len(CLASS_NAMES),
                                       preprocessing=preprocessing)

  #* 1/3: Validate model's base prediction accuracy (about 97%)
  print('[TASK 1/3] Validate orginal prediction:')
  utils.validate(fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE)

  #* 2/3: Perform an adversarial attack with blackbox attack
  print('[TASK 2/3] Generate adversaries:')
  #TODO: attack methods accepts: HopSkipJumpAttack, SinglePixelAttack
  attack = attack_switcher(ATTACK_METHOD, fmodel)

  tic = time.time()
  pbar = tqdm(dataset_loader)
  pbar.set_description('Generate adversaries')

  # iterate through images to generate adversaries
  adversaries = []
  for image, label in pbar:
    #TODO: may need to change according to attack method
    adv = attack(image.numpy(), label.numpy())

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
