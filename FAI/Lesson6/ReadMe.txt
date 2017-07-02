Note for Lesson 6 folder:

Spent a lot of time during this lesson figuring out why I wasn't able to replicate lecture results for an RNN built in Theano.

The reason turned out to be the line:
  `from __future__ import division`
This imports the Python3 version of division (floating point) and uses that instead of the Python2 version (integer). So my SGD optimizer was stuck with 2÷3 = 0 instead of 2÷3 = 0.666... which is why it wasn't able to get beyond a very shallow local minima in its loss function.

Part of solving this issue involved me going on a wild-goose-chase, thinking that a previous model being run had altered the input data array and thus the output of the Theano-built RNN. I replicated the changes and they had no effect ~ I'm no sure they even mattered when the data was formatted for an categorical RNN.

Theano_RNN_test_3 contains a successful working version of code among irrelevant code.
Theano_RNN_test_3-Copy1_Success distills the above to just the relevant parts.
Theano_RNN_Solved shows the difference due to the division import, and wraps up this exercise.

- WH Nixalo 2017
