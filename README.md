<img src="https://i.loli.net/2020/02/26/SAx6ejplMbQovNG.png" align="right" width="125px" height="auto" alt="f00l!">

<img src="https://i.loli.net/2020/02/26/gi5bjUaqshn4du9.png" width="150px" height="auto" alt="F00L"/>

<h6>The Robustness of Adversarial Examples</h6>

![](https://img.shields.io/badge/uses-PyTorch-5b8c85?logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/python-3.7.6-297ca0?logo=python&logoColor=white)
![](https://img.shields.io/badge/lab-Jupyter-ec9b3b?logo=Jupyter&logoColor=white)
![](https://img.shields.io/badge/GPU-2080Ti-76B900?logo=NVIDIA&logoColor=white)
![](https://img.shields.io/badge/license-MIT-434e52)

<h2>Table of contents</h2>

- [Experiment overview](#experiment-overview)
  - [Training and evaluation dataset](#training-and-evaluation-dataset)
  - [Target models](#target-models)
  - [Attack methods (non-targeted)](#attack-methods-non-targeted)
  - [Perturbation budget](#perturbation-budget)
  - [Image scaling](#image-scaling)
- [Structure](#structure)
- [Building and running](#building-and-running)

> This is my final year project. I am investigating the impact of image scaling on the effectiveness of adversarial attacks targeted on ConvNets (or other neural networks).

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

- FGSM (`GradientSignAttack`)
- BIM (`LinfinityBasicIterativeAttack`)
- MIM (`MomentumIterativeAttack`)
- DeepFool (`DeepFoolLinfinityAttack`)
- CW (`CarliniWagnerL2Attack`)

**Black-box attack:**

- HopSkipJumpAttack - Boundary Attack++ (`HopSkipJumpAttack`)
- GenAttack (`GenAttack`) *Needs to be initialized with a target, in our case: `9`.*

### Perturbation budget

|           |   ![](https://latex.codecogs.com/svg.latex?\ell_p)    |     Attacks     | Level 1 | Level 2 | Level 3 | Level 4 |
| :-------: | :---------------------------------------------------: | :-------------: | :-----: | :-----: | :-----: | :-----: |
| White box | ![](https://latex.codecogs.com/svg.latex?\ell_\infty) | FGSM, BIM, MIM  |  4/255  |  8/255  | 16/255  | 32/255  |
| White box |   ![](https://latex.codecogs.com/svg.latex?\ell_2)    |  DeepFool, C&W  |    2    |    4    |    6    |    8    |
| Black box | ![](https://latex.codecogs.com/svg.latex?\ell_\infty) | HSJA, GenAttack | 64/255  | 72/255  | 80/255  | 88/255  |

### Image scaling

- Control Group (×1)
- Nearest-neighbor: `interpolation=cv2.INTER_NEAREST` (×2 | ×0.5)
- Bi-linear: `interpolation=cv2.INTER_LINEAR` (×2 | ×0.5)
- Pixel area relation: `interpolation=cv2.INTER_AREA` (×2 | ×0.5)
- Bi-cubic: `interpolation=cv2.INTER_CUBIC` (×2 | ×0.5)
- Lanczos: `interpolation=cv2.INTER_LANCZOS4` (×2 | ×0.5)

## Structure

| Directory                                        | Purpose                                                                                                                                                                                                            |
| :----------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`convnet_trainer.ipynb`](convnet_trainer.ipynb) | Creating a state-of-the-art ConvNet by transfer training default ConvNet models that PyTorch provides against the ImageNette dataset                                                                               |
| [./utils](./utils)                               | Utility module for `convnet_trainer.ipynb` to use when training different models with different architectures.                                                                                                     |
| [./adversarial_attack](./adversarial_attack)     | Try to implement Steps 2-3 as shown in the experiment overview                                                                                                                                                     |


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

**Most of the valuable and more useful code** is inside the folder `adversarial_attack`.

**Most pre-trained weights** have been uploaded to Google Drive for you to use right away. Dig into the folders' README for more info.

If you see a Jupyter Notebook and a Python Script which share the same name, **always use the notebook!**

---

📦 **F00L: Investigating the Robustness of Adversarial Examples** ©Spencer Woo.

Released under the [MIT License](./LICENSE).

Authored and maintained by Spencer Woo.

[@Portfolio](https://spencerwoo.com/) · [@Blog](https://blog.spencerwoo.com/) · [@GitHub](https://github.com/spencerwooo)
