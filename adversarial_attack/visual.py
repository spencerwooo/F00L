#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize results from saved *.npy files.

Files:
  * Images (and adversarial examples):
    `*.npy`: visualize with the first two cells.
  * L2/L∞ norms:
    `*.npy`: scatter plot with the last few cells.

"""

# %%
import numpy as np
import matplotlib.pyplot as plt


def img_to_np(image):
  return np.transpose(image, (1, 2, 0))


# %%
adv = np.load('adv.npy')
img = np.load('img.npy')
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
dist_fgsm = np.load('dist_fgsm.npy')
dist_deep_fool = np.load('dist_deep_fool.npy')
dist_jsma = np.load('dist_jsma.npy')
indice = np.arange(0, 100, 1)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(indice, dist_fgsm, label='fgsm, l2')
ax.scatter(indice, dist_deep_fool, label='deep fool, l2')
ax.scatter(indice, dist_jsma, label='jsma, l2')
ax.set_ylim(0, 8)
ax.set_ylabel('L2 norm distance')
ax.set_xlabel('Image index')
ax.legend()
plt.title('ResNet adversaries perturbation size (l2 norm)')
plt.show()
