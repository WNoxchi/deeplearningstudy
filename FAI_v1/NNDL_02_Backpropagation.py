# from Book: Neural Networks and Deep Learning - Chapter 02
# 2017-May-07 11:28

class Network(object):
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch. The
        "mini_batch" is a list of typles "(x, y)", and "eta" is
        the learning rate."""
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

# Most of the work is done by the line delta_nabla_b, delta_nabla_w =
# self.backprop(x, y)
    def backprop(self, x, y):
        """Return a tuple ("nabla_b, nabla_w" representing the gradient
        for the cost function C_x. "nabla_b" and "nabla_w" are layer-
        by-layer lists of numpy arrays, similar to "self.biases"
        and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]
        # FeedForward
        activation = x
        activations = [x] # list to store all activations, layer-by-layer
        zs = [] # list to store all z vectors, layer-by-layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1]], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # NOTE that the variable 1 in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here,
        # 1 = 1 means the last layer of neurons, 1 = 2 is the
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for λ in xrange(2, self.num_layers):
            z = zs[-λ]
            zp = sigmoid_prime(z)
            delta = np.dot(self.weights[-λ+1].transpose(), delta) * sp
            nabla_b[-λ] = delta
            nabla_w[-λ] = np.dot(delta, activations[-λ-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives δC_x/δa
        for the output activations."""
        return (output_activations - y)

def sigmoid(z):
    """The sigmoid functions."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid functions."""
    return sigmoid(z) * (1 - sigmoid(z))
