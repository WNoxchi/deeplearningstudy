# Wayne H Nixalo - 2017-Jul-01 14:28
# FAI01 - Practical Deep Learning I - Lesson 7
# code from vgg16bn.py. See https//github.com/fastai/courses/nbs

# Recurrent Neural Network in Python

#------------# Setup Basic functions

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_d(x):
    output = sigmoid(x)
    return output*(1-output)

def relu(x): return np.maximum(0., x)
def relu_d(x): return (x > 0.)*1.

# self check
relu(np.array([3., -3.])), relu_d(np.array([3., -3.]))

# euclidean distance & derivative of
def dist(a,b): return pow(a-b,2)
def dist_d(a,b): return 2*(a-b)

import pdb

# cross entropy & derivative of
eps = 1e-7
def x_entropy(pred, actual):
    # clip preds otws 1s & 0s create ∞s & destroy everything
    return -np.sum(actual * np.log(np.clip(pred, eps, 1-eps))
def x_entropy_d(pred, actual): return -actual/pred

def softmax(x): return np.exp(x)/np.exp(x).sum()
def softmax_d(x):
    sm = softmax(x)
    res = np.expand_dims(-sm,-1)*sm
    res[np.diag_indices_from(res)] = sm*(1-sm)
    return res

#------------# checking PyRNN results match TheanoRNN results

test_preds = np.array([0.2, 0.7, 0.1])
test_actuals = np.array([0., 1., 0.])
nnet.categorical_crossentropy(test_preds, test_actuals).eval()

test_grad(test_preds)

x_entropy_d(test_preds, test_actuals)

pre_pred = random(oh_x_rnn[0][0].shape)
preds = softmax(pre_pred)
actual = oh_x_rnn[0][0]
np.allclose(softmax_d(pre_pred).dot(loss_d(preds,actual)), preds-actual)

softmax(test_preds)

nnet.softmax(test_preds).eval()

test_out = T.flatten(nnet.softmax(test_inp))
test_grad = theano.function([test_inp], theano.gradient.jacobian(test_out, test_inp))
test_grad(test_preds)

softmax_d(test_preds)

act=relu
act_d = relu_d
loss=x_entropy
loss_d = x_entropy_d

def scan(fn, start, seq):
    res = []
    prev = start
    for s in seq:
        app = fn(prev, s)
        res.append(app)
        prev = app
    return res

scan(lambda prev, curr: prev+curr, 0, range(5))

#------------# Setup Up Training

# 8 char seqs 1Hencoded
inp = oh_x_rnn
# 8 char seqs ea. moved across by 1; & 1Hencoded
outp = oh_y_rnn
# voc size = 86 chars
n_input = vocab_size
n_output = vocab_size

inp.shape, outp.shape
# ((75110, 8, 86), (75110, 8, 86))

def one_char(prev, item):
    # Previous state
    tot_loss, pre_hidden, pre_pred, hidden, ypred = prev
    # Current inputs and output
    x, y = item
    pre_hidden = np.dot(x,w_x) + np.dot(hidden,w_h)
    hidden = act(pre_hidden)
    pre_pred = np.dot(hidden,w_y)
    ypred = softmax(pre_pred)
    return (
        # Keep track of loss so we can report it
        tot_loss+loss(ypred, y),
        # Used in backprop
        pre_hidden, pre_pred,
        # Used in next iteration
        hidden,
        # To provide predictions
        ypred)

def get_chars(n): return zip(inp[n], outp[n])

# we first need to do the forward pass
# this is currently not doing it statefully -- it's passing a vector of zeros;
# to make it stateful, we'd just have the final state returned by this, then
# feed it back the next time through the loop.
# NOTE: you probably won't get great results doing that bc when you do things
# statefully you're much more likely to have gradients & activations explode
# unless you do a GRU or LSTM
def one_fwd(n): return scan(one_char, (0,0,0,np.zeros(n_hidden),0), get_chars(n))
# ^ scan thru all chars in nth phrase (inp & outp), calling the one_char function

# "Columnify" a vector - required for weight updates
def col(x): return x[:,newaxis]

# backward pass for gradient descent
def one_bkwd(args, n):
    global w_x, w_y, w_h
    # we grab one of our inputs, one of our outputs, then we go backwards,
    # through each one, ea. of the 8 chars, from the end to start
    i = inp[n]  # 8x86
    o = outp[n] # 8x86
    d_pre_hidden = np.zeros(n_hidden) # 256
    for p in reversed(range(len(i))):
        totloss, pre_hidden, pre_pred, hidden, ypred = args[p]
        x=i[p] # 86
        y=o[p] # 86
        d_pre_pred = softmax_d(pre_pred).dot(loss_d(ypred,y))   # 86
        # the derivative of a matrix multiplication is the multiplication with
        # the transpose of that matrix | also, the hidden weight matrix has
        # 2 arrows going into it and 2 going out: so we have to add those
        # derivatives | finally we have to 'undo' the activation function, so
        # multiply it by the derivative of the activation function --- and
        # that's the chain rule that gets us back to the start weight matrix.
        d_pre_hidden = (np.dot(d_pre_hidden, w_h.T)
                        + np.dot(d_pre_pred, w_y.T)) * act_d(pre_hidden) # 256

        # d(loss)/d(w_y) = d(loss)/d(pre_pred) * d(pre_pred)/d(w_y)
        w_y -= col(hidden) * d_pre_pred * alpha
        # d(loss)/d(w_h) = d(loss)/d(pre_hidden[p-1]) * d(pre_hidden[p-1])/d(w_h)
        if (p>0): w_h -= args[p-1][3].dot(d_pre_hidden) * alpha
        w_x -= col(x)*d_pre_hidden * alpha
    return d_pre_hidden

scale = math.sqrt(2./n_input)
w_x = normal(scale=scale, size=(n_input,n_hidden))
w_y = normal(scale=scale, size=(n_hidden,n_output))
w_h = np.eye(n_hidden, dtype=np.float32)

# here is our loop
overallError = 0
alpha = 0.0001
for n in xrange(10000):
    res = one_fwd(n)
    overallError +== res[-1][0]
    deriv = one_bkwd(res,n)
    if(n % 1000 == 999):
        # if(True):
        print("Error:{:.4f}; Gradient:{:.5f}".format(overallError/1000,
                                                     np.linalg.norm(deriv)))
        overallError = 0
