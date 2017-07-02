# Wayne H Nixalo - 2017-Jul-01 20:57
# FAI01 - Practical Deep Learning I

# I've been having a problem getting a RNN built in Theano to work. A corpus of
# Nietzsche is the training data. Done correctly, the model should start with
# a loss of ~25 and ends at ~14.4, and reasonably predict the next character.
# Done wrong, the model starts with a loss ~30~29, and ends at ~25, and
# predicts only empty spaces (obvious easy local minima).

# I've narrowed down the relevant parts of code, going on a goose-hunt pursuing
# red herrings, until finally discovering the model works as advertised when
# copied, but not when I rewrite it. So this is to see where I made errors and
# how they're responsible.

# NOTE: ohhh my holy fuck. The culprit was the:
#           from __future__ import division, print_function
# line. Specifically `import division`. That single import is responsible for
# the last 2 weeks of tracking down this issue. So why? Well w/o looking into
# it, it seems like somewhere integer division was supposed to be done where
# floating-point div was instead done or vice-versa.

# NOTE: okay got it. importing division from __future__ gives you Python3
# divison: floating-point. My poor RNN's SGD optimizer was forced to use
# integer division everywhere instead of floating-point. Oof. Well, that's done.



# COPIED THEANO RNN MODEL:
######## INITIAL SETUP #########################################################

import theano
%matplotlib inline
import sys, os
sys.path.insert(1, os.path.join('../utils'))
import utils; reload(utils)
from utils import *
from __future__ import division, print_function


path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read()
print('corpus length:', len(text))


chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
print('total chars:', vocab_size)


chars.insert(0, "\0")
# ''.join(chars[1:-6])


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))



idx = [char_indices[c] for c in text]
# the 1st 10 characters:
# idx[:10]

######## DATA FORMATTING #######################################################

cs = 8 # use 8 characters to predict the 9th

c_in_dat = [[idx[i+n] for i in xrange(0, len(idx)-1-cs, cs)] for n in range(cs)]
xs = [np.stack(c[:-2]) for c in c_in_dat]

c_out_dat = [[idx[i+n] for i in xrange(1, len(idx)-cs, cs)] for n in range(cs)]
ys = [np.stack(c[:-2]) for c in c_out_dat]

oh_ys = [to_categorical(o, vocab_size) for o in ys]
oh_y_rnn = np.stack(oh_ys, axis=1)

oh_xs = [to_categorical(o, vocab_size) for o in xs]
oh_x_rnn = np.stack(oh_xs, axis=1)

oh_x_rnn.shape, oh_y_rnn.shape

######## THEANO RNN ############################################################

n_hidden = 256; n_fac = 42; cs = 8

n_input = vocab_size
n_output = vocab_size

def init_wgts(rows, cols):
    scale = math.sqrt(2/rows) # 1st calc Glorot number to scale weights
    return shared(normal(scale=scale, size=(rows, cols)).astype(np.float32))
def init_bias(rows):
    return shared(np.zeros(rows, dtype=np.float32))
def wgts_and_bias(n_in, n_out):
    return init_wgts(n_in, n_out), init_bias(n_out)
def id_and_bias(n):
    return shared(np.eye(n, dtype=np.float32)), init_bias(n)

# Theano variables
t_inp = T.matrix('inp')
t_outp = T.matrix('outp')
t_h0 = T.vector('h0')
lr = T.scalar('lr')

all_args = [t_h0, t_inp, t_outp, lr]

W_h = id_and_bias(n_hidden)
W_x = wgts_and_bias(n_input, n_hidden)
W_y = wgts_and_bias(n_hidden, n_output)
w_all = list(chain.from_iterable([W_h, W_x, W_y]))

def step(x, h, W_h, b_h, W_x, b_x, W_y, b_y):
    # Calculate the hidden activations
    h = nnet.relu(T.dot(x, W_x) + b_x + T.dot(h, W_h) + b_h)
    # Calculate the output activations
    y = nnet.softmax(T.dot(h, W_y) + b_y)
    # Return both (the 'Flatten()' is to work around a theano bug)
    return h, T.flatten(y, 1)

[v_h, v_y], _ = theano.scan(step, sequences=t_inp,
                            outputs_info=[t_h0, None], non_sequences=w_all)

error = nnet.categorical_crossentropy(v_y, t_outp).sum()
g_all = T.grad(error, w_all)

def upd_dict(wgts, grads, lr):
    return OrderedDict({w: w-g*lr for (w,g) in zip(wgts,grads)})

upd = upd_dict(w_all, g_all, lr)

# we're finally ready to compile the function!:
fn = theano.function(all_args, error, updates=upd, allow_input_downcast=True)

X = oh_x_rnn
Y = oh_y_rnn
X.shape, Y.shape

err=0.0; l_rate=0.01
for i in xrange(len(X)):
    err += fn(np.zeros(n_hidden), X[i], Y[i], l_rate)
    if i % 1000 == 999:
        print ("Error:{:.3f}".format(err/1000))
        err=0.0


#################################################################################
# REWRITTEN THEANO RNN MODEL:
######## INITIAL SETUP #########################################################

import theano
# %matplotlib inline
import sys, os
sys.path.insert(1, os.path.join('../utils'))
# import utils; reload(utils)
from utils import *
# from __future__ import division, print_function

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
print('total chars:', vocab_size)

chars.insert(0, "\0")

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

idx = [char_indices[c] for c in text]

######## DATA FORMATTING #######################################################

cs = 8

c_in_dat = [[idx[i+n] for i in xrange(0, len(idx)-1-cs, cs)] for n in xrange(cs)]
xs = [np.stack(c[:-2]) for c in c_in_dat]

c_out_dat = [[idx[i+n] for i in xrange(1, len(idx)-cs, cs)] for n in xrange(cs)]
ys = [np.stack(c[:-2]) for c in c_out_dat]

oh_ys = [to_categorical(o, vocab_size) for o in ys]
oh_y_rnn = np.stack(oh_ys, axis=1)

oh_xs = [to_categorical(o, vocab_size) for o in xs]
oh_x_rnn = np.stack(oh_xs, axis=1)

oh_x_rnn.shape, oh_y_rnn.shape

######## THEANO RNN ############################################################

n_hidden = 256; n_fac = 42; cs = 8

n_input = vocab_size
n_output = vocab_size

def init_wgts(rows, cols):
    scale = math.sqrt(2/rows)
    return shared(normal(scale=scale, size=(rows,cols)).astype(np.float32))
def init_bias(rows):
    return shared(np.zeros(rows, dtype=np.float32))
def wgts_and_bias(n_in, n_out):
    return init_wgts(n_in, n_out), init_bias(n_out)
def id_and_bias(n):
    return shared(np.eye(n, dtype=np.float32)), init_bias(n)

# Theano Variables
t_inp = T.matrix('inp')
t_outp = T.matrix('outp')
t_h0 = T.vector('h0')
lr = T.scalar('lr')

all_args = [t_h0, t_inp, t_outp, lr]

W_h = id_and_bias(n_hidden)
W_x = wgts_and_bias(n_input, n_hidden)
W_y = wgts_and_bias(n_hidden, n_output)
w_all = list(chain.from_iterable([W_h, W_x, W_y]))

def step(x, h, W_h, b_h, W_x, b_x, W_y, b_y):
    # Calculate hidden activations
    h = nnet.relu(T.dot(x, W_x) + b_x + T.dot(h, W_h) + b_h)
    # Calculate output activations
    y = nnet.softmax(T.dot(h, W_y) + b_y)
    # Return both   --   `flatten()` is Theano bug workaround
    return h, T.flatten(y, 1)

[v_h, v_y], _ = theano.scan(step, sequences=t_inp,
                            outputs_info=[t_h0, None], non_sequences=w_all)
error = nnet.categorical_crossentropy(v_y, t_outp).sum()
g_all = T.grad(error, w_all)

def upd_dict(wgts, grads, lr):
    return OrderedDict({w: w - g * lr for (w, g) in zip(wgts, grads)})

upd = upd_dict(w_all, g_all, lr)

# ready to compile the function:
fn = theano.function(all_args, error, updates=upd, allow_input_downcast=True)

X = oh_x_rnn
Y = oh_y_rnn
# X.shape, Y.shape

# semi-auto SGD loop:
err=0.0; l_rate=0.01
for i in xrange(len(X)):
    err += fn(np.zeros(n_hidden), X[i], Y[i], l_rate)
    if i % 1000 == 999:
        print ("Error: {:.3f}".format(err/1000))
        err=0.0
