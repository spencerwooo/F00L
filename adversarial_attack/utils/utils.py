import os
import time

import foolbox
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import cv2


def load_trained_model(model_name=None, model_path='', class_num=10):
  # supported models: 'resnet', 'vgg', 'inception', 'mobilenet'
  model = None

  # load models
  if model_name == 'resnet':
    model = torchvision.models.resnet18(pretrained=True)
    # for param in model.parameters():
    #   param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, class_num)

  elif model_name == 'vgg':
    model = torchvision.models.vgg11(pretrained=True)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, class_num)

  elif model_name == 'inception':
    model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, class_num)

  elif model_name == 'mobilenet':
    model = torchvision.models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, len(CLASS_NAMES))

  else:
    raise NotImplementedError('Model not supported')

  model.load_state_dict(torch.load(model_path))
  model.eval()
  return model


def load_dataset(dataset_path=None, dataset_image_len=1, batch_size=4):
  # resize image to size 213 * 213
  transform = transforms.Compose(
      [transforms.Resize((213, 213)),
       transforms.ToTensor()])

  class_start_indice = [indice * 200 for indice in range(0, dataset_image_len)]
  images_in_class_indice = np.array(
      [[j for j in range(k, k + dataset_image_len)]
       for k in class_start_indice]).flatten()

  # load dataset with validation images
  dataset = torchvision.datasets.ImageFolder(root=dataset_path,
                                             transform=transform)

  # 1. get 10 images from 10 classes for a total of 100 images, or ...
  dataset = torch.utils.data.Subset(dataset, images_in_class_indice)
  # 2. get first 100 images (all tenches)
  # dataset = torch.utils.data.Subset(dataset, range(0, 100))

  # compose dataset into dataloader
  # (don't shuffle, no need to shuffle, we're not training.)
  dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  # get dataset size (length)
  dataset_size = len(dataset)

  print('Loaded data from: {} with a total of {} images.'.format(
      dataset_path, dataset_size))

  return dataset_loader, dataset_size


def validate(fmodel, dataset_loader, dataset_size, batch_size=4, advs=None):
  pbar = tqdm(dataset_loader)

  # if adv is default (None), validate predictions
  stage = 'predictions' if advs is None else 'adversaries'

  pbar.set_description('Validate {}'.format(stage))
  pbar.set_postfix(acc='0.0%')

  preds = []
  acc = 0.0
  for i, (image, label) in enumerate(pbar):
    # make a prediction on either original dataset or adversaries
    if advs is None:
      prob = fmodel.forward(image.numpy())
    else:
      prob = fmodel.forward(advs[i])

    pred = np.argmax(prob, axis=-1)
    preds.append(pred)

    # calculate current accuracy
    acc += np.sum(pred == label.numpy())
    current_acc = acc * 100 / ((i + 1) * batch_size)
    pbar.set_postfix(acc='{:.2f}%'.format(current_acc))

  acc = acc * 100 / dataset_size
  pbar.write('Validated {} with accuracy of: {:.2f}%'.format(stage, acc))
  return preds


def notify(time_elapsed, notify='notify.py'):
  """
  Send notifications
  """

  bitjs = '~/.net/BIT.js'
  title = 'Attack complete'
  msg = 'Time elapsed {:.2f}m {:.2f}s'.format(time_elapsed // 60,
                                              time_elapsed % 60)
  cmd = 'python {} -b "{}" -t "{}" -m "{}"'.format(notify, bitjs, title, msg)
  stream = os.popen(cmd)
  output = stream.read()
  print('\n' + output)


def scale_adv(advs, resize_scale, interpolation_method):
  """
  Resize adversaries with 5 different methods
  """
  interpolation_methods = {
      'INTER_NEAREST': cv2.INTER_NEAREST,
      'INTER_LINEAR': cv2.INTER_LINEAR,
      'INTER_AREA': cv2.INTER_AREA,
      'INTER_CUBIC': cv2.INTER_CUBIC,
      'INTER_LANCZOS4': cv2.INTER_LANCZOS4
  }
  interpolation = interpolation_methods[interpolation_method]
  resized_advs = []

  # default adversaries are stored in 4 images each batch
  for adv_batch in advs:
    resized_adv_batch = []
    for adv in adv_batch:
      resized_adv = cv2.resize(np.moveaxis(adv, 0, 2), (0, 0),
                               fx=resize_scale,
                               fy=resize_scale,
                               interpolation=interpolation)
      resized_adv_batch.append(np.moveaxis(resized_adv, 2, 0))
    resized_advs.append(np.array(resized_adv_batch))

  print('Image scaling done! Resized advs using {} with a scale of {}.'.format(
      interpolation_method, resize_scale))
  return resized_advs
