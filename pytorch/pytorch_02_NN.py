# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# 2017-Oct-30 01:42
from time import time; t0 = time()

# Neural Networs constructed using the torch.nn package
# nn depends on autograd to define models and differentiate them.
# an nn.Module contains layers, and a method forward(input) that returns output

# DEFINE THE NETWORK
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# the above defines the forward function; the backward fn where the grads are
# computed is automatically defd for you using autograd. You can use any of the
# Tensor ops in the forward fn.

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

# Input to the forward is an autogra.Variable, and so is the output.
# NOTE: expected input size to this net(LeNet) is 32x32. To use on MNIST, resize
#       images to 32x32.
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# NOTE: torch.nn only supports mini-batches. The entire torch.nn package only
#       supports inputs that are a mini-batch of samples, not a single sample.
#       nn.Conv2d will take a 4D Tensor of nSamples x nChannels x Height x Width
#       If you have a single sample, just use input.unsqueeze(0) to add a fake
#       batch dim.

# LOSS FUNCTION
output = net(input)
target = Variable(torch.arange(1,11))   # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# if you follow loss in the backward direction using it's .grad_fn attrib,
# you'll see a graph of computations that looks like this:
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss

# So, when we call loss.backward(), the whole graph is differentiated wrt the
# loss, and all Variables in the graph will have their .grad Variable
# accumulated w/ the gradient.
# Following a step backward:
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU

# BACKPROP:
# to backpropagate the error: just need to call loss.backward(). Need clear the
# existing gradients, or else grads will be accumulated to existing grads.

# Calling loss.backward() and looking at conv1's bias grads before/after backward:
net.zero_grad() # zeroes the gardient buffers of all parameters

print('conv1.bias/grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# UPDATE THE WEIGHTS
# the simplest update rule: SGD: weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# torch.optim allows you to use different update rules
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

tTime = time() - t0;
print("Total running time of script: {:.6f} seconds".format(tTime))
