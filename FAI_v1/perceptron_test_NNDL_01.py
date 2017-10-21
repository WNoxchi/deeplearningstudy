# Zhayna: Neural Networks and Deep Learning Ch01
# Wayne H Nixalo - 2017-May-03 11:35

import numpy as np

def vec(x):
    return np.array([i for i in x])

def pcn(x, b, c=1):
    w = np.array([1 for i in range(len(x))])
    return np.dot(c*w, x) + (c * b)
    # return (c * w).dot(x) + (c * b)

x = np.array([2, 1])
b = 10
c = 1313
b = 5.1

def pnet(x, b, c=1):
    λ1 = [pcn(x, b, c) for i in range(3)]
    print("λayer 1:",λ1)
    λ2 = [pcn(λ1, b, c) for i in range(2)]
    print("λayer 2:",λ2)
    y = pcn(λ2, b, c)
    return int(y > 0)

for i in range(6):
    c = 100*i**3 + 1
    print("output: ",pnet(x, b, c))
    print("scalar: ",c)
    print()


#NOTE: changing the scaling factor in a perceptron network will
# not change the output unless w•x+b = 0. The scalar will only
# push the network further in the direction the dot Π was already
# taking it. ie: -5 —> -500; still < 0. If w•x + b = 0 ==> the
# output will flip once the dotΠ nolonger equals 0.

# # OUTPUT:
# λayer 1: [8.0999999999999996, 8.0999999999999996, 8.0999999999999996]
# λayer 2: [29.399999999999999, 29.399999999999999]
# output:  1
# scalar:  1
#
# λayer 1: [818.09999999999991, 818.09999999999991, 818.09999999999991]
# λayer 2: [248399.39999999999, 248399.39999999999]
# output:  1
# scalar:  101
#
# λayer 1: [6488.1000000000004, 6488.1000000000004, 6488.1000000000004]
# λayer 2: [15594989.4, 15594989.4]
# output:  1
# scalar:  801
#
# λayer 1: [21878.099999999999, 21878.099999999999, 21878.099999999999]
# λayer 2: [177292019.39999998, 177292019.39999998]
# output:  1
# scalar:  2701
#
# λayer 1: [51848.099999999999, 51848.099999999999, 51848.099999999999]
# λayer 2: [995671709.39999998, 995671709.39999998]
# output:  1
# scalar:  6401
#
# λayer 1: [101258.10000000001, 101258.10000000001, 101258.10000000001]
# λayer 2: [3797546279.4000001, 3797546279.4000001]
# output:  1
# scalar:  12501
