import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    x = sigmoid(x)
    return x * (1 - x)

class MLP():
    def __init__(self, input_size, output_size, layers_size):
        self.weights_1 = np.random.random((layers_size, input_size)) 
        self.weights_2 = np.random.random((output_size, layers_size)) 
        self.output_size = output_size

    def forward(self, x):
        u1 = np.matmul(x, self.weights_1.T)
        a1 = sigmoid(u1)
        
        u2 = np.matmul(a1, self.weights_2.T)
        a2 = sigmoid(u2)

        return a2

    def loss(self, y, predict):
        return (y - predict)**2


network = MLP(2, 1, 3)
y_hat = network.forward(np.array([1,2]))
loss = network.loss([1], y_hat)
print( loss)


