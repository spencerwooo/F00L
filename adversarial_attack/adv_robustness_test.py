"""
Run five types of image scaling algorithms and re-validate adversaries.
"""

import torch
import foolbox
import numpy as np

from utils import utils

# load saved adversaries
ADV_SAVE_PATH = 'advs/hop_skip_jump_attack_adv.npy'
# pretrained model state_dict path
MODEL_NAME = 'resnet'
MODEL_PATH = '../models/resnet_foolbox/200224_0901_resnet_imagenette.pth'
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


def init_models(model_name, model_path):
  model = utils.load_trained_model(model_name=model_name,
                                   model_path=model_path,
                                   class_num=len(CLASS_NAMES))
  print('Instantiated pretrained {}.'.format(MODEL_NAME))
  return model


def main():
  # load model
  model = init_models(MODEL_NAME, MODEL_PATH)
  # define preprocessing procedures (Foolbox)
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
  # load adversaries
  advs = np.load(ADV_SAVE_PATH)

  #* TASK 1/3: validate original adversaries
  print('\n[TASK 1/3] Validate original adversaries:')
  utils.validate(fmodel,
                 dataset_loader,
                 dataset_size,
                 batch_size=BATCH_SIZE,
                 advs=advs)

  #* TASK 2/3: resize adversaries
  print('\n[TASK 2/3] Resize adversaries:')
  scales = [0.5, 2]
  methods = [
      'INTER_NEAREST', 'INTER_LINEAR', 'INTER_AREA', 'INTER_CUBIC',
      'INTER_LANCZOS4'
  ]
  # initialize resized adversaries
  resized_advs = {
      method: {scale: None
               for scale in scales}
      for method in methods
  }
  for method in methods:
    for scale in scales:
      resized_advs[method][scale] = utils.scale_adv(advs, scale, method)

  #* TASK 3/3: validate resized adversaries
  print('\n[TASK 3/3] Validate resized adversaries:')
  for method in methods:
    for scale in scales:
      print('[Resize method: {:<14}, scale factor: {:>3}]'.format(
          method, scale))
      utils.validate(fmodel,
                     dataset_loader,
                     dataset_size,
                     batch_size=BATCH_SIZE,
                     advs=resized_advs[method][scale])


if __name__ == "__main__":
  main()
