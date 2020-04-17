# Adversaries generated

This directory stores the adversaries generated during adversarial attacks. Download pre-generated attack adversaries following the link below.

## Directory structure

You should organize the `.npy` files under this directory according to the following structure:

```
.
├── README.md
├── resnet
│   ├── bim
│   │   └── 0417_0708_0.031_adv.npy
│   ├── cw
│   │   └── 0417_0710_5.000_adv.npy
│   ├── df
│   │   └── 0417_0708_0.031_adv.npy
│   ├── fgsm
│   │   └── 0417_0546_0.031_adv.npy
│   ├── ga
│   │   └── 0417_0936_0.500_adv.npy
│   ├── hsj
│   │   └── 0416_0906_0.031_adv.npy
│   └── mim
│        └── 0417_0708_0.031_adv.npy
└── mobilenet
│   └── ...
└── vgg
│   └── ...
└── inception
     └── ...
```

## List of files

### `resnet`

#### White box

| File name                                  | Google Drive Link                                                              |
| :----------------------------------------- | :----------------------------------------------------------------------------- |
| `advs/resnet/fgsm/0417_0546_0.031_adv.npy` | [FGSM](https://drive.google.com/open?id=1E-JAbJ7D9eyx1P2JSKX4Jst8ek6aGrdB)     |
| `advs/resnet/bim/0417_0708_0.031_adv.npy`  | [BIM](https://drive.google.com/open?id=1fhKFpnZ51uKeg6uZR5ciRh1BdwP1I9-V)      |
| `advs/resnet/mim/0417_0708_0.031_adv.npy`  | [MIM](https://drive.google.com/open?id=14OxhUCl6CO6EPx_0FWZGLTg8-gk49pl1)      |
| `advs/resnet/cw/0417_0710_5.000_adv.npy`   | [CW](https://drive.google.com/open?id=1yA-G8_JX6mxisko3gwPWz90PgxvgGkAi)       |
| `advs/resnet/df/0417_0708_0.031_adv.npy`   | [DeepFool](https://drive.google.com/open?id=15kTue-FlzatDi8EqY8-S32Mvo2x5G1Ms) |

#### Black box

| File name                                 | Google Drive Link                                                                 |
| :---------------------------------------- | :-------------------------------------------------------------------------------- |
| `advs/resnet/hsj/0416_0906_0.031_adv.npy` | [HopSkipJump](https://drive.google.com/open?id=1JWQ6lHMg_yKo_AZX6noI1n12y8V4xQVw) |
| `advs/resnet/ga/0417_0936_0.500_adv.npy`  | [GenAttack](https://drive.google.com/open?id=1n7UtpXx_LxLlNArI2XQs7eG1ffbqTVI6)   |

### `mobilenet`

### `vgg`

### `inception`
