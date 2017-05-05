# Neural Networks for Deep Learning                     Python2.7
# Chapter 01: Using Neural Networks to recognize handwritten digits
# http://neuralnetworksanddeeplearning.com/chap1.html <-- code from here too.
# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

# NOTE Python2 may not like non-ASCII chars but I stand by my λ's.

# Wayne H. Nixalo - 2017-May-03 20:50
# TNWK-0501:Kawkasos

import random
import numpy as np

# Network class used to represent a neural network:
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # num nodes in respective layers
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # Aldashna: creating Network object w/ 2 nodes 1st layer, 3 in second, 1 final:
    # net = Network([2,3,1])

    # NumPy matrix storing weights connecting 2nd & 3rd layers of nodes:
    # net.weights[1]

    # Code computing the output from a Network instance. Begin by defining the
    # sigmoid function:
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # NOTE: when z vec or Numpy array NP auto applies sigmoid elemwise (veczd form)
    # Input assumed to be (n, 1) Numpy ndarray, not (n,) vector.
    # Next add a FeedForward method to Network class. Given input, a, for network,
    # rets corresp output. Just applies Eqn<22>: a' = σ(wa + b) for ea. layer.
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # Stochastic Gradient Descent method to allow our Network to learn:
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        The "training_data" is a list of tuples "(x, y)" representing the training
        inputs and the desired output. The other non-optional parameters are
        self-explanatory. If "test_data" is provided then the network will be
        evaluated against the test data after each epoch, and partial progress
        printed out. This is useful for tracking progress, but slows things down
        substantially"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch. The "mini_batch" is a list
        of typles "(x, y", and "eta" is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    # most of the work is done by the line:
    #   delta_nabla_b, delta_nabla_w = self.backprop(x, y)

################################################################################
    # ##### code below covered in Chapter 02:
    def backprop(self, x, y):
        """Return a typle ``(nabla_b, nabla_w)`` representing the gradient for the
        cost function C_x. ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
        numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the avitvations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # NOTE that the variable 1 in the loop below is used a little differently
        # to the notation in Chapter 2 of the book. Here, 1 = 1 means the last
        # layer of neurons, 1 = 2 is the second-last layer, and so on. It's a
        # renumbering of the scheme in the book, used here to take avantage of the
        # fact that Python can use negative indicies in lists.
        for λ in xrange(2, self.num_layers):
            z = zs[-λ]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-λ+1].transpose(), delta) * sp
            nabla_b[-λ] = delta
            nabla_w[-λ] = np.dot(delta, activations[-λ - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network
        outputs the correct result. Note that the neural network's output
        is assumed to be the index of whichever neuron in the final layer
        has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives δC_x / δa
        for the ouput activations."""
        return (output_activations - y)

### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

















#
