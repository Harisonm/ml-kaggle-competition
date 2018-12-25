import numpy as np
from numpy.linalg import inv

X = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [4]
])

W = (inv(X.transpose().dot(X)).dot(X.transpose())).dot(Y)

print(W)

print(W.flatten().transpose().dot(X[2]))