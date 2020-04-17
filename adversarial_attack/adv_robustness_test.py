"""
Run five types of image scaling algorithms and re-validate adversaries.
"""

import foolbox
import numpy as np
import torch
from tqdm import tqdm

from utils import utils

# Models: resnet / vgg / mobilenet / inception
MODEL_NAME = "resnet"
# Modify according to adversary file relative path
ADV_SAVE_PATH = "advs/resnet/fgsm/0417_0546_0.031_adv.npy"

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
DATASET_IMAGE_NUM = 10
DATASET_PATH = "../data/imagenette2-160/val"


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


def main():
  """ Validate adv -> Rescale -> Validate scaled adv """

  model = init_models(MODEL_NAME)
  preprocessing = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
  )

  dataset_loader, dataset_size = utils.load_dataset(
    dataset_path=DATASET_PATH, dataset_image_len=DATASET_IMAGE_NUM
  )

  # use GPU if available
  if torch.cuda.is_available():
    model = model.cuda()

  fmodel = foolbox.models.PyTorchModel(
    model,
    bounds=(0, 1),
    num_classes=len(CLASS_NAMES),
    preprocessing=preprocessing,
  )

  advs = np.load(ADV_SAVE_PATH)

  # * TASK 1/3: validate original adversaries
  control_group_acc = utils.validate(
    fmodel, dataset_loader, dataset_size, batch_size=BATCH_SIZE, advs=advs
  )

  # * TASK 2/3: resize adversaries
  scales = [0.5, 2]
  methods = [
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LANCZOS4",
  ]
  # Initialize resized adversaries dict
  resized_advs = {
    method: {scale: None for scale in scales} for method in methods
  }

  pbar = tqdm(total=len(scales) * len(methods), desc="SCL")
  for method in methods:
    for scale in scales:
      resized_advs[method][scale] = utils.scale_adv(advs, scale, method)
      pbar.update(1)
  pbar.close()

  # * TASK 3/3: validate resized adversaries
  print(
    "{:<19} - success: {}%".format("Control group", 100 - control_group_acc)
  )
  for method in methods:
    for scale in scales:
      acc = utils.validate(
        fmodel,
        dataset_loader,
        dataset_size,
        batch_size=BATCH_SIZE,
        advs=resized_advs[method][scale],
        silent=True,
      )
      print("{:<14} ×{:<3} - success: {}%".format(method, scale, 100 - acc))


if __name__ == "__main__":
  main()
