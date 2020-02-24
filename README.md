<div align="center">
  <img src="./assets/banner-adv.png" width="600px" height="auto" alt="adversarials" />

  <h1>Adversarial Attacks</h1>

  🎃 *Generating adversarial examples to trick the neural network.*

  ![](https://img.shields.io/badge/python-3.7.6-4381b2?logo=python&logoColor=white&style=flat-square)
  ![](https://img.shields.io/badge/using-PyTorch-ee4c2c?logo=PyTorch&logoColor=white&style=flat-square)
  ![](https://img.shields.io/badge/license-MIT-black?&style=flat-square)

 </div>

<h6>This is my final year project. I am investigating the impact of image scaling on the effectiveness of adversarial attacks targeted on ConvNets (or other neural networks).</h6>

## Experiment overview

**Step 1:** Using PyTorch to load train / validation datasets, transfer train ConvNets for further experiments.

**Step 2:** Using Foolbox to generate adversarial examples in order to trick the ConvNets into mis-classifying images from the validation dataset.

**Step 3:** Using different interpolation algorithms to scale adversarials and measure the impact of this transformation on the effectiveness of the attack on different models.

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

- FGSM
- DeepFool
- JSMA (SaliencyMapAttack)
- CW
- MI-FGSM (MomentumIterativeAttack)

**Black-box attack:**

- Single Pixel Attack
- HopSkipJumpAttack (Boundary Attack++)

### Image scaling

- Control Group (×1)
- Nearest-neighbor: `interpolation=cv2.INTER_NEAREST` (×2, ×0.5)
- Bi-linear: `interpolation=cv2.INTER_LINEAR` (×2, ×0.5)
- Pixel area relation: `interpolation=cv2.INTER_AREA` (×2, ×0.5)
- Bi-cubic: `interpolation=cv2.INTER_CUBIC` (×2, ×0.5)
- Lanczos: `interpolation=cv2.INTER_LANCZOS4` (×2, ×0.5)

## Structure

|              Directory               | Purpose                                                                                                                                                                                                                                                   |
| :----------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|           [./misc](./misc)           | All are sample scripts where I follow examples from official PyTorch / Keras tutorials and try to recreate the same results.                                                                                                                              |
| [./resnet_foolbox](./resnet_foolbox) | <ul><li>Creating a ResNet18 ConvNet by transfer training the default ResNet18 model against the ImageNette dataset.</li><li>Attacking the trained ConvNet with Foolbox, measuring the effectiveness of said attack after image transformations.</li></ul> |
|    [./vgg_foolbox](./vgg_foolbox)    | <ul><li>Creating a VGG11 ConvNet by transfer training the default VGG11 model against the ImageNette dataset.</li><li>Attacking the trained ConvNet with Foolbox, measuring the effectiveness of said attack after image transformations.</li></ul>       |
|                 ...                  | ...                                                                                                                                                                                                                                                       |

## Building and running

Create conda environment by file `environment.yml`:

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

Add dependencies in `environment.yml`, and run the following command to install them:

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

🎃 **Final Year Project** ©Spencer Woo. Released under the [MIT License](./LICENSE).

Authored and maintained by Spencer Woo.

[@Portfolio](https://spencerwoo.com/) · [@Blog](https://blog.spencerwoo.com/) · [@GitHub](https://github.com/spencerwooo)
