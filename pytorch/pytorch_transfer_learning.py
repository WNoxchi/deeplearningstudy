# 2018-Jan-13 12:12
# http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# TRANSFER LEARNING TUTORIAL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os

plt.ion() # interactive mode
