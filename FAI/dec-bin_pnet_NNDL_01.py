# Zhayna: Neural Networks and Deep Learning Ch01
# Wayne H Nixalo - 2017-May-03 11:35 - 19:49

# Find a set of weights and biases for an output layer than converts a decimal
# digit into binary. Assume the correct output in the decimal output layer has
# activate ≥ 0.99 and incorrect outputs have ≤ 0.01

import numpy as np
from numpy import linalg as LA

x = [[int(r==c) for r in range(10)] for c in range(10)]
y = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],
     [0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]]
w = [[0 for r in range(len(y))] for c in range(len(y[0]))]
x = np.array(x)
y = np.array(y)

for b in range(len(w)):
    for d in range(len(w[b])):
        if y[d][b] == 1:
            w[b][d] = 10
        else:
            w[b][d] = 1e-3
w = np.array(w)

def output_test(x):
    y = [0,0,0,0]
    for i in range(len(x)):
        for k in range(len(y)):
            y[k] += w[k][i] * x[i]
    for k in range(len(y)):
        if y[k] < 0.01:
            y[k] = 0
        elif y[k] > 0.99:
            y[k] = 1
    return y

for i in x:
    print(output_test(i))

# for r in x:
#     print(r)
#
# for r in y:
#     print(r)
#
# for r in w:
#     print(r)


# This is a reverse mapping of Decimal to Binary
# x = [1 0 0 0 0 0 0 0 0 0]
#     [0 1 0 0 0 0 0 0 0 0]
#     [0 0 1 0 0 0 0 0 0 0]
#     [0 0 0 1 0 0 0 0 0 0]
#     [0 0 0 0 1 0 0 0 0 0]
#     [0 0 0 0 0 1 0 0 0 0]
#     [0 0 0 0 0 0 1 0 0 0]
#     [0 0 0 0 0 0 0 1 0 0]
#     [0 0 0 0 0 0 0 0 1 0]
#     [0 0 0 0 0 0 0 0 0 1]
#
# y = [0 0 0 0]
#     [0 0 0 1]
#     [0 0 1 0]
#     [0 0 1 1]
#     [0 1 0 0]
#     [0 1 0 1]
#     [0 1 1 0]
#     [0 1 1 1]
#     [1 0 0 0]
#     [1 0 0 1]
#
# w = [0 0 0 0 0 0 0 0 1 1]
#     [0 0 0 0 1 1 1 1 0 0]
#     [0 0 1 1 0 0 1 1 0 0]
#     [0 1 0 1 0 1 0 1 0 1]






################################################################################
# # Fallout of my first attempt:
#
# # for r in x: print(r)
# # print()
# # for r in y: print(r)
# # print()
# # for r in w: print(r)
#
# for row in range(len(w)):
#     for col in range(len(w[row])):
#         if y[col][row] == 1:
#             w[row][col] = 10
#         else:
#             w[row][col] = -10
#
# # for r in w: print(r)
#
#
# W = [[1,1,0,0,0,0,0,0,0,0],
#      [0,0,1,1,1,1,0,0,0,0],
#      [0,0,1,1,0,0,1,1,0,0],
#      [1,0,1,0,1,0,1,0,1,0]]
# for r in range(len(W)):
#     for c in range(len(W[r])):
#         if W[r][c] == 1:
#             W[r][c] = 10
#         elif W[r][c] == 0:
#             W[r][c] = -5
#
#
#
# def vec(x):
#     return np.array([i for i in x])
#
# def pcn(x, b=0, c=1, w='1'):
#     if w == '1': w = np.array([1 for i in range(len(x))])
#     return np.dot(c*w, x) + (c * b)
#     # return (c * w).dot(x) + (c * b)
#
# # Find a set of weights and biases for an output layer than converts a decimal
# # digit into binary. Assume the correct output in the decimal output layer has
# # activate ≥ 0.99 and incorrect outputs have ≤ 0.01
#
# # Hand-coded method: create answer set, use PLA - perceptron learning algorithm
# # to compute the weights. Update: w1 = w0 + x*y; y: scalar
#
# dec = 10
# tol = 1e-2
#
# x = np.array([[int(row == col) for row in range(dec)] for col in range(dec)])
# y = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],
#      [0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]]
# # w = [[1. for row in range(len(y))] for col in range(len(y[0]))]
# w = [[1. for row in range(len(x))] for col in range(len(y[0]))]
# b = np.array([1. for i in range(len(y[0]))]) # b is 1x4
#
# w = np.hstack((np.reshape(b,(4,1)),w)) # w is now 4x11, w[:,0] = b[:]
# # print(w)
# x = np.hstack((np.reshape(np.ones(10),(10,1)),x)) # x is now 10x11, x[:,0] = b[:]
#
# # for r in range(len(x)):
# #     for c in range(len(x[r])):
# #         if x[r][c] == 1:
# #             x[r][c] = 0.999
# #         else:
# #             x[r][c] = 0.001
# #
# # for r in range(len(y)):
# #     for c in range(len(y[r])):
# #         if y[r][c] == 1:
# #             y[r][c] = 0.999
# #         else:
# #             y[r][c] = 0.001
# # print(y)
#
# # print(x)
# # print(y)
# print(w)
#
# def check(w,x,y):
#     """
#     Determines Weights & Biases via Perceptron Learning Algorithm:
#     If W[α]•X is not within tolerance of y[α] W[α] is updated via:
#         W[α] = {W[α] + y * X : y ≥ .99; W[α] - y * X : y ≤ .01}
#     """
#     change = True
#     for _ in range(10):
#     # while change:
#     #     change = False
#         # for each output node (4)
#         for α in range(len(y[0])):
#             # for each penultimate node (10)
#             for i in range(len(y)):
#                 # perform update-check
#                 # print(np.dot(w[α],x[i]) - y[i][α])
#                 # print("before:     ", w[α])
#                 if LA.norm(np.dot(w[α],x[i]) - y[i][α]) > tol:
#                     if y[i][α] > 0.99:
#                         # print("w[α]:\n", w[α])
#                         # print(" - ", -1. * x[i])
#                         w[α] += -1. * x[i]
#                         # print("w2[α]: \n", w[α])
#                         # print("α: {}, i:{}".format(α, i))
#                         # print("\n")
#                         # print(w[α])
#                     # else:
#                     elif y[i][α] < 0.01:
#                         w[α] +=  1. * x[i]
#                         # print(w[α])
#                     change = True
#     return w
#
# W_k = check(w, x, y)
# print(W_k)
#
# for i in range(len(x)):
#     print(np.dot(W_k, x[i]))
#
# # for i in range(len(w)):
# #     print(w[i])
# #     print()
