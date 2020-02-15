# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
  """
  Define a CNN neural network for CIFAR10 image classification
  """

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


def gpu_init():
  """ Initialize GPU """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  return device


def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


device = gpu_init()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load CIFAR 10 dataset
train_set = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
# get some random training images
data_iter = iter(train_loader)
images, labels = data_iter.next()

# show images and print labels
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %%
# initialize CNN
net = Net()
# tranfer CNN onto GPU training
net.to(device)

# define classification Cross-Entropy loss and SGD with momentum
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
# train the network
tic = time.clock()
for epoch in range(2):

  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get inputs and labels, data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward propaganda + backward propaganda + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print training statistics
    running_loss += loss.item()
    if i % 2000 == 1999:
      print('[Epoch %d - batch: %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

toc = time.clock()
print('Finished training with a time of %.2fs.' % (toc - tic))

# %%
# examine the test images
data_iter = iter(test_loader)
images, labels = data_iter.next()

# print image from the test dataset
imshow(torchvision.utils.make_grid(images))
print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# predict images
outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# %%
# examine prediction accuracy of all classes
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# predict images in test dataset
with torch.no_grad():

  # iterate through test images
  for data in test_loader:
    images, labels = data[0].to(device), data[1].to(device)

    # predict images
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

    # evaluate prediction accuracy
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1

for i in range(10):
  print('Accuracy of %5s : %2d %%' %
        (classes[i], 100 * class_correct[i] / class_total[i]))
