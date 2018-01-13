# 1.  TENSORS

# 1.1 WARM-UP: NUMPY

import numpy as np

# N is batch size; D_in is input dim
# H is hidden dim; D_out is output dim
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(y, loss)

    # Backprop to compute gradients of w1 and w2 wrt loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# 1.2 PYTORCH: TENSORS

# To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype.
import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # uncomment to run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 & w2 wrt loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# 2.  AUTOGRAD

# 2.1 PYTORCH: VARIABLES AND AUTOGRAD

# Using PyTorch Variables and Autograd to implement a two-layer network; Now we
# no longer need to manually implement the backward pass through the network:

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # uncomment to run on GPU

# N is batch size; D_in is input dim; H hidden dim; D_out output dim
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input & outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we don't need to compute gradients
# wrt these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(troch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Variables: these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we don't need to keep references to intermediate values
    # since we're not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Variables.
    # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
    # (1,); loss.data[0] is a scalar vaue holding the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # Use Autograd to compute the backward pass. This call will compute the
    # gradient of loss wrt all Variables with requires_grad=True.
    # After this call w1.grad & w2.grad will be Variables holding the gradient
    # of the loss wrt w1 & w2 respectively.
    loss.backward()

    # Update weights using gradient descent; w1.data & w2.data are Tensors,
    # w1.grad & w2.grad are Variables, and w1.grad.data & w2.grad.data
    # are Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()


# 2.2 PYTORCH: DEFINING NEW AUTOGRAD FUNCTIONS

# Defining a custom Autograd function for performing ReLU, and using it to
# implement our 2-layer network:
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom Autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. You can cache arbitrary Tensors for use
        in the backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss wrt the output, and we need to compute the gradient of the loss
        wrt the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# batch size, in dim, hidden dim, out dim
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input & outputs, wrap them in Variables
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Construct an instance of our MyReLU class to use in our network
    relu = MyReLU()

    # Forward pass: compute predicted y using oeprations on Variables; we
    # compute ReLU using ur custom autograd operation
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(y, loss.data[0])

    # Use Autograd to compute the backward pass
    loss.backward()

    # Update weights using gradient descent
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating the weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()


# 2.3 TENSORFLOW: STATIC GRAPHS

# USING TENSORFLOW INSTEAD OF PYTORCH TO FIT A SIMPLE 2-LAYER NET:
# (STATIC VS DYNAMIC COMPUTATION GRAPHS)
import tensorflow as tf
import numpy as np

# First we set up the computational graph:

# batch size, in dim, hidden dim, out dim
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# NOTE that this code doesn't actually perform any numeric operations; it
# merely sets up the computational graph that we'll later execute.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h,, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss wrt w1 & w2
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. NOTE that
# in TensorFlow the act of updating the value of the weights is part of the
# copmutational graph; inPyTorch this happens outside the computational graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we've built our comptuational graph, so we enter a TensorFlow session
# to actually execute the graph.
with tf.Session() as sess:
    # Run the graph once to initialize the Variables w1 and w2.
    sess.run(tf.global_variables_initializer())

    # Create NumPy arrays holding the actual data for inputs x and targets y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for _ in range(500):
        # Execute the graph many times. Each time it executes we want to bind
        # x_value to x and y_value to y, specified with the feed_dict argument.
        # Each time we execute the graph we want to compute the values for
        # loss, new_w1, and new_w2; the values of these Tensors are returned as
        # NumPy arrays.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)

# 3.  NN MODULE

# 3.1 PYTORCH: NN

# The nn package defines a set of Modules which are roughly equivalent to
# network layers. We use nn package to implement our 2-layer network:
import torch
from torch.autograd import Variable

# batch size, in dim, hidden dim, out dim
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
                            torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),
                            )

# The nn package also contains definitions of popular loss functions; in this
# case we'll use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module
    # objects override the __call__ operator so you can call them like
    # functions. When doing so you pass a Variable of input data to the Modeul
    # and it produces a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and
    # true values of y, and the loss function returns a Variable containing
    # the loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss wrt all the learnable
    # parameters of the model. Internally, the parameters of each Module are
    # stored in Variables with requires_grad=True, so this call will compute
    # gradients for all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable,
    # so we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data


# 3.2 PYTORCH: OPTIM

# Optimizing the model using the Adam algorithm in the optim package
import torch
from torch.autograd import Variable

# bs in hid out
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model and loss function
model = torch.nn.Sequential(
                            torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out),
                            )
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that'll update the weights of
# the model for us. Here we'll use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it'll update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: copmute gradient of the loss wrt model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its pars
    optimizer.step()


# 3.3 PYTORCH: CUSTOM N MODULES

# Implementing 2-Layer Network as Custom Module subclass
import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them
        as member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.lienar1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# bs in hid out
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable paramters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.paramters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Copmute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3.4: PYTORCH: CONTROL FLOW + WEIGHT SHARING   ################################

import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we'll
        use in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.moddile_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2,
        or 3 and reuse the middle_linear Module that many times to compute
        hidden layer representations.

        Since each forward pass builds a dynamic computation graph, we can use
        normal Python control-flow operators like loops or conditional
        statements when defining the forward pass of the model.

        Here we also see that it's perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement
        from Lua Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

# bs in hid out
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instatitating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this straing model
# with vanilla stochastic gradient descent is touch, so we use momentum.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momenum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()















































#
