<h1>The Robustness of Adversarial Examples</h1>

![](https://img.shields.io/badge/using-PyTorch-ee4c2c?logo=PyTorch&logoColor=white&style=flat-square)
![](https://img.shields.io/badge/python-3.7.6-4381b2?logo=python&logoColor=white&style=flat-square)
![](https://img.shields.io/badge/built%20with-Jupyter-f37626?logo=Jupyter&logoColor=white&style=flat-square)
![](https://img.shields.io/badge/license-MIT-black?&style=flat-square)

<h6>This is my final year project. I am investigating the impact of image scaling on the effectiveness of adversarial attacks targeted on ConvNets (or other neural networks).</h6>

## Table of contents

- [Table of contents](#table-of-contents)
- [Experiment overview](#experiment-overview)
  - [Training and evaluation dataset](#training-and-evaluation-dataset)
  - [Target models](#target-models)
  - [Attack methods (non-targeted)](#attack-methods-non-targeted)
  - [Image scaling](#image-scaling)
- [Structure](#structure)
- [Building and running](#building-and-running)

## Experiment overview

**Step 1:** Using PyTorch to load train/validation datasets, transfer train ConvNets for further experiments.

**Step 2:** Using Foolbox to generate adversarial examples to trick the ConvNets into misclassifying images from the validation dataset.

**Step 3:** Using different interpolation algorithms to scale adversaries and measure the impact of this transformation on the effectiveness of the attack on different models.

### Training and evaluation dataset

[ImageNette](https://github.com/fastai/imagenette): a smaller subset of 10 easily classified classes from ImageNet. (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute)

You can download ImageNette training and validation images here: [160 px download](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz).

### Target models

- ResNet18
- VGG11
- MobileNet V2
- Inception V3

### Attack methods (non-targeted)

**White-box attack:**

- FGSM (GradientSignAttack)
- DeepFool (DeepFoolAttack)
- JSMA (SaliencyMapAttack)
- CW (CarliniWagnerL2Attack)
- MI-FGSM (MomentumIterativeAttack)

**Black-box attack:**

- Single Pixel Attack
- HopSkipJumpAttack (Boundary Attack++)

### Image scaling

- Control Group (×1)
- Nearest-neighbor: `interpolation=cv2.INTER_NEAREST` (×2 | ×0.5)
- Bi-linear: `interpolation=cv2.INTER_LINEAR` (×2 | ×0.5)
- Pixel area relation: `interpolation=cv2.INTER_AREA` (×2 | ×0.5)
- Bi-cubic: `interpolation=cv2.INTER_CUBIC` (×2 | ×0.5)
- Lanczos: `interpolation=cv2.INTER_LANCZOS4` (×2 | ×0.5)

## Structure

| Directory                                                                     | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [./misc](./misc)                                                              | All are sample scripts where I follow examples from official PyTorch / Keras tutorials and try to recreate the same results.                                                                                                                                                                                                                                                                                                                                                               |
| [./resnet_foolbox](./resnet_foolbox)<br>[./vgg_foolbox](./vgg_foolbox)<br>... | 1. Creating a state-of-the-art ConvNet by transfer training default ConvNet models that PyTorch provides against the ImageNette dataset: `xxx_imagenette_transfer.ipynb`<br>2.Attacking the ConvNet with FGSM only, visualize generated adversaries: `xxx_fgsm_attack_sample.ipynb`<br>3. Attacking the trained ConvNet with [FGSM, DeepFool, JSMA, CW, MI_FGSM] using Foolbox, measuring the effectiveness of said attack after image transformations: `xxx_adv_with_image_scaling.ipynb` |


## Building and running

Create a new conda environment by file `environment.yml`:

```bash
conda env create --file environment.yml
```

Enter conda environment `adv`:

```bash
conda activate adv
```

Run jupyter lab in a browser.

```bash
jupyter lab
```

If you wish to install other dependencies, please add them manually in `environment.yml`, and run the following command to install them:

```bash
conda env update
```

<h6>See here on <a href="https://stackoverflow.com/questions/39280638/how-to-share-conda-environments-across-platforms">How to share conda environments across platforms</a></h6>

Exit conda environment:

```bash
conda deactivate
```

**Most of the valuable and more useful code** is in folders named in the `*_foolbox` pattern.

**Most pre-trained weights** have been uploaded to Google Drive for you to use right away. Dig into the Jupyter Notebooks for more info.

If you see a Jupyter Notebook and a Python Script which share the same name, **always use the notebook!**

---

🎃 **The Robustness of Adversarial Examples** ©Spencer Woo. Released under the [MIT License](./LICENSE).

Authored and maintained by Spencer Woo.

[@Portfolio](https://spencerwoo.com/) · [@Blog](https://blog.spencerwoo.com/) · [@GitHub](https://github.com/spencerwooo)
