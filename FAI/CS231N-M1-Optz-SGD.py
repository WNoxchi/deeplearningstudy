import numpy as np
# assume X_train is the data where each column is an example (eg. 3073 x 50,000)
# assume Y_train are the labels (eg. 1D array of 50,000)
# assume the finction L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
    W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
    loss = L(X_train, Y_train, W) # get the loss over the entire training set
    if loss < bestloss: # keep track of the best solution
        bestloss = loss
    print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
