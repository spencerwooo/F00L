# -*- coding: utf-8 -*-

# %%
from __future__ import print_function

import torch
# %%
print(torch.cuda.get_device_name(0))
print('Trying out PyTorch.')

# %%
x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([5.5, 3])
print(x)
