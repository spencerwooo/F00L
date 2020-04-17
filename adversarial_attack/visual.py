#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize results from saved *.npy files.

Files:
  * Images (and adversarial examples):
    `*.npy`: visualize with the first two cells.
  * L2/L∞ norms:
    `*.npy`: scatter plot with the last few cells.

"""

# pylint: disable=invalid-name
# %%
import numpy as np
import matplotlib.pyplot as plt


def img_to_np(image):
  """ Transpose image to viewable format to plot/visualize. """
  return np.transpose(image, (1, 2, 0))


# %%
adv = np.load("adv.npy")
img = np.load("img.npy")
pert = adv - img
_l2 = np.linalg.norm(pert)

# %%
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img_to_np(img).squeeze())
plt.subplot(1, 3, 2)
plt.imshow(img_to_np(adv).squeeze())
plt.subplot(1, 3, 3)
plt.imshow(img_to_np(adv - img).squeeze())

# %%
THRESHOLD = 8 / 255
NORM = "inf"

# dist_fgsm = np.load("dist_fgsm.npy")
# dist_deep_fool = np.load("dist_deep_fool.npy")
dist_hsj = np.load("dist_hsj.npy")
# dist_ga = np.load("dist_ga.npy")

dist = dist_hsj

# %%
length = 100
indice = np.arange(0, length, 1)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
# ax.scatter(indice, dist_fgsm, label="fgsm, l2")
# ax.scatter(indice, dist_deep_fool, label="deep fool, l2")

ax.scatter(indice, dist, label="HopSkipJumpAttack, l{}".format(NORM))

ax.set_ylabel("l{} norm distance".format(NORM))
ax.set_xlabel("adversaries")
ax.legend()

plt.hlines(y=THRESHOLD, xmin=0, xmax=length, colors="r")
plt.ylim(0, THRESHOLD * 2)
plt.title("ResNet adversaries perturbation size (l{} norm)".format(NORM))
plt.show()

print(
  "L_{}: min {:.4f}, mean {:.4f}, max {:.4f}".format(
    NORM, dist.min(), np.median(dist), dist.max()
  )
)
