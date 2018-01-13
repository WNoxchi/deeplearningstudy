# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# 2017-Oct-30 02:12
# WNixalo

# TRAINING A CLASSIFIER

# Generally for data you can use std python packages that load data into a
# NumPy Array. Then convert this array into a torch.*Tensor
#   * For images: Pillow & OpenCV are useful
#   * For audio: scipy & librosa
#   * For text: either raw Python or Cython based loading, or NLTK & SpaCy

# torchvision is specfly created for vision; has data loaders for common data-
# sets: ImageNet, CIFAR10, MNIST, etc. and data trsfmrs for images, viz.,
# torchvision.datasets, and torch.utils.data.DataLoader

# this tutorial will use the CIFAR10 dataset. CIFAR-10 imgs are size 3x32x32
# (10 classes)

# TRAINING AN IMAGE CLASSIFIER
# 1. Load and Normalize the CIFAR10 training/test datsets using torchvision
# 2. Define a Convolutional Neural Network
# 3. Define a Loss Function
# 4. Train the network on the trainind data
# 5. Test the network on the test data

# 1. LOADING AND NORMALIZING CIFAR10
# Using torch vision:
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision datasets output are PILImage images of range[0,1]. We trsfm them
# to Tensors of normalized range[-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Showing some of the training images:
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2. DEFINE A CONVOLUTIONAL NEURAL NETWORK
# modifying the NN form the NN section to take 3-channel images instead of 1
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
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


net = Net()

# 3. DEFINE A LOSS FUNCTION AND OPTIMIZER
# using classification cross entropy loss & sgd w/ momentum
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. TRAIN THE NETWORK
# loop over the data iterator, and feed the inputs to the network & optimize:
for epoch in range(2):  # loop over datset multipl times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 5. TEST THE NETWORK ON THE TEST DATA
# checking predicted class label against ground-truth
# displaying an image from the test set:
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# now running the network on these:
outputs = net(Variable(images))

# The outputs are energies for the 10 classes. The higher the more the network
# thinks that image is that particular class. Getting index of highest energy:
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Looking at how network performs on the entire dataset:
correct = 0; total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Chance is 10%. Viewing the classes the network did well on and not:
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# TRAINING ON GPU:
# You transfer a Neural Net to the GPU the same way you trsfr a Tensor. This'll
# recursively go over all modules and convert their parameters and buffers to
# CUDA tensors:
net.cuda()

# Remember you'll have to send inputs & targets at every step to the GPU too:
inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

# the bigger this network (it's v.small r.now) the greater the speedup.

















#
