"""
Run five types of image scaling algorithms and re-validate adversaries.

Image interpolation methods:

  * INTER_NEAREST
  * INTER_LINEAR
  * INTER_AREA
  * INTER_CUBIC
  * INTER_LANCZOS4

! Best if ran inside VS Code's Python Interactive panel.
! (for better plotting experiences.)
"""

# %%
# pylint: disable=invalid-name
import csv
import os

import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from matplotlib.lines import Line2D
from tqdm import tqdm

from utils import utils

# Models: resnet, vgg, mobilenet, inception
# Attacks: fgsm, bim, mim, df, cw, hsj, ga
TARGET_MODEL = "resnet"
ATTACK_METHOD = "fgsm"
BUDGET_LEVEL = 1

SAVE_RESULTS = True
PLOT_RESULTS = True

ADV_SAVE_PATH = "advs/{}/{}/adv_level{}.npy".format(
  TARGET_MODEL, ATTACK_METHOD, BUDGET_LEVEL
)

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
PLOT_SAVE_PATH = os.path.join("robust_plots", TARGET_MODEL)
PLOT_SAVE_NAME = "{}_level{}_robust".format(ATTACK_METHOD, BUDGET_LEVEL)


def init_models():
  """ Initialize pretrained CNN models """

  model_path = {
    "resnet": MODEL_RESNET_PATH,
    "vgg": MODEL_VGG_PATH,
    "mobilenet": MODEL_MOBILENET_PATH,
    "inception": MODEL_INCEPTION_PATH,
  }

  loaded_model = utils.load_trained_model(
    model_name=TARGET_MODEL,
    model_path=model_path.get(TARGET_MODEL),
    class_num=len(CLASS_NAMES),
  )
  return loaded_model


def save_results_csv(data):
  """ Save results to csv file for further analysis """

  if SAVE_RESULTS:
    fields = list(data.keys())
    fields.insert(0, "ATTACK_METHODS")
    data["ATTACK_METHODS"] = "{}_level{}".format(ATTACK_METHOD, BUDGET_LEVEL)

    csv_file_path = "results/{}.csv".format(TARGET_MODEL)
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a+") as f:
      writer = csv.DictWriter(f, fieldnames=fields)
      if not file_exists:
        writer.writeheader()
      writer.writerow(data)
    data.pop("ATTACK_METHODS", None)


def plot_results(original_data, flattened_data):
  """ Plot the attack success rates of different adversaries. """

  if PLOT_RESULTS:
    keys = list(flattened_data.keys())
    data = list(flattened_data.values())

    avg_05 = np.average(list(original_data[0.5].values()))
    avg_2 = np.average(list(original_data[2].values()))

    rcParams["font.family"] = "monospace"

    cmap = plt.cm.coolwarm
    color_val = {1: cmap(0.9), 0.5: cmap(0.25), 2: cmap(0.05)}
    colors = []
    for key in flattened_data:
      if "1" in key:
        colors.append(color_val[1])
      if "0.5" in key:
        colors.append(color_val[0.5])
      if "2" in key:
        colors.append(color_val[2])

    custom_lgd = [
      Line2D([0], [0], color=color_val[1], lw=4),
      Line2D([0], [0], color=color_val[0.5], lw=4),
      Line2D([0], [0], color=color_val[2], lw=4),
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(keys, data, color=colors, width=0.3)
    plt.hlines(avg_05, 1, 5, colors=color_val[0.5], linestyles="dashed")
    plt.hlines(avg_2, 6, 10, colors=color_val[2], linestyles="dashed")

    plt.annotate(
      "{}%".format(data[0]),
      xy=(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height()),
      color=color_val[1],
      xytext=(0, 3),
      textcoords="offset points",
      ha="center",
      va="bottom",
    )
    plt.text(1, avg_05 + 8, "avg: {}%".format(avg_05), color=color_val[0.5])
    plt.text(6, avg_2 + 8, "avg: {}%".format(avg_2), color=color_val[2])

    plt.xticks(rotation="25", ha="right")
    plt.ylabel("attack success rate (%)")
    plt.ylim(0, np.max(data) * 1.2)
    plt.legend(custom_lgd, ["control group", "scale ×0.5", "scale ×2"])
    plt.title(
      "{}: {} level {} adversary robustness".format(
        TARGET_MODEL, ATTACK_METHOD, BUDGET_LEVEL
      )
    )
    plt.tight_layout()

    # save plot to local
    if not os.path.exists(PLOT_SAVE_PATH):
      os.makedirs(PLOT_SAVE_PATH)
    plt.savefig(
      os.path.join(PLOT_SAVE_PATH, PLOT_SAVE_NAME), dpi=100,
    )
    plt.show()


# %%
# Validate adv -> Rescale -> Validate scaled adv

model = init_models()
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
resized_advs = {method: {scale: None for scale in scales} for method in methods}

pbar = tqdm(total=len(scales) * len(methods), desc="SCL")
for method in methods:
  for scale in scales:
    resized_advs[method][scale] = utils.scale_adv(advs, scale, method)
    pbar.update(1)
pbar.close()

# * TASK 3/3: validate resized adversaries
print(
  "{:<19} - success: {}%".format("CONTROL_GROUP  ×1", 100 - control_group_acc)
)

# Initialize success rate data
success_data = {1: {"CONTROL_GROUP": 100.0 - control_group_acc}, 0.5: {}, 2: {}}
success_data_flatten = {"CONTROL_GROUP ×1": 100.0 - control_group_acc}

for scale in scales:
  for method in methods:
    acc = utils.validate(
      fmodel,
      dataset_loader,
      dataset_size,
      batch_size=BATCH_SIZE,
      advs=resized_advs[method][scale],
      silent=True,
    )
    success_data[scale][method] = 100.0 - acc
    success_data_flatten["{} ×{}".format(method, scale)] = 100.0 - acc
    print("{:<14} ×{:<3} - success: {}%".format(method, scale, 100.0 - acc))

save_results_csv(success_data_flatten)

# %%
# * Plot results (success rate - advs)
plot_results(success_data, success_data_flatten)
