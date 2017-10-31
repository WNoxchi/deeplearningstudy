# 2017-Oct-30 01:20   WNixalo     PyTorch tutorial
# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# Autograd: Automatic Differentiation

# pytorch autograd is a define-by-run framework: meaning your backprop is
# defined by how your code is run, and that every single iteration can be different.

# VARIABLE
# autograd.Variable wraps a Tensor and supports nearly all ops defd on it.
# once you finish your computation you can call .backward() and have all the
# gradients computed automatically.
#
# you can access the raw tensor thru the .data attribute, while the gradient
# wrt this variable is accumulated into .grad

# The Variable and Function classes are interconnected and build up an acyclic
# graph that encodes a complete history of computation. Each variable has a
# .grad_fn attrib tht refs a Function that's created the Variable;
# Variables created by the user have a grad_fn of None.
#
# To compute derivatives, call .backward() on a Variable. If Variable scalar
# dont need specfy args. Otws need specfy grad_output arg thats tensor of
# matching shape.

import torch
from torch.autograd import Variable

# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

# Do an operation on variable:
y = x + 2
print(y)
# y was created as a result of an op --> so it has a grad_fn
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)

# GRADIENTS
# out.backward() is equivalent to doing out.backward(torch.Tensor([1.0]))
out.backward()
# print gardients ∆(out)/∆x
print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)















#
