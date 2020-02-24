# -*- coding: utf-8 -*-

# %%
import foolbox
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# import resnet18 for attack
resnet18 = models.resnet18(pretrained=True).eval()

# use GPU if available
if torch.cuda.is_available():
  resnet18 = resnet18.cuda()

# instantiate foolbox model
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
# using pretrained model: resnet18
fmodel = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1),
                                     num_classes=1000,
                                     preprocessing=(mean, std))

# get a few image examples from imagenet (20 available)
# images are retrieved from index 1 to 5 (batchsize is 5)
BATCH_SIZE = 6
INDEX = 7

images, labels = foolbox.utils.samples(dataset='imagenet',
                                       batchsize=BATCH_SIZE,
                                       index=INDEX,
                                       data_format='channels_first')

# normalize images
images = images / 255

# %%
# real work begins
PLT_ROW = 3
PLT_COL = 2


def make_predictions(images, labels, plot_title):
  """ Predict input category and plot image """
  num = BATCH_SIZE

  # plot image and predictions
  plt.figure(figsize=(8 * PLT_COL, 4 * PLT_ROW))

  # make predictions on all images in a batch
  probs = fmodel.forward(images)

  # iterate through the batch of images
  i = 0
  for (image, label, prob) in zip(images, labels, probs):

    # image category prediction
    pred = np.argmax(prob)

    # top 8 prediction labels
    top_pred_labels = np.argpartition(prob, -8)[-8:]
    # probabilities of the top 8 predictions
    top_pred_prob = [(prob[label])
                     for label in top_pred_labels]

    # list of top 8 prediction labels
    # (converted into strings for matplot to use as its axis)
    label_list = list(map(str, top_pred_labels))

    # prediction success or fail
    success = (pred == label)

    # plot target image
    plt.subplot(PLT_ROW, PLT_COL * 2, i * 2 + 1)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title(plot_title)

    # plot predictions probability
    plt.subplot(PLT_ROW, PLT_COL * 2, i * 2 + 2)
    # green (#36b37e): success
    # red (#ff3333): failure
    color = ('#36b37e' if success else '#ff3333')
    # define background bar color
    this_plot = plt.barh(label_list,
                         top_pred_prob,
                         height=0.45,
                         color="#c1c7d0")

    # define success bar / failure bar color
    this_plot[label_list.index(str(pred))].set_color(color)

    # define title
    plt.title('Ground truth: {} / Prediction: {}'.format(label, pred))

    i = i + 1

  plt.show()


# make predictions on all original images
make_predictions(images, labels, 'Original image')

# %%
# construct attack model
attack = foolbox.v1.attacks.FGSM(fmodel)

# iterate through images to generate adversarial examples
adv_list = []
for (image, label) in zip(images, labels):
  adversarial = attack(image, label)
  adv_list.append(adversarial)

# organize adversarial examples into np.array
adversarials = np.array(adv_list)

# %%
# make predictions on adversarial examples
make_predictions(adversarials, labels, 'Adversarial example')
