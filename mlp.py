import numpy as np
import math
import random

def relu(x):
    return max(0.0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    x = sigmoid(x)
    return x * (1 - x)

def derivative_relu(x):
    relu_grad = x > 0
    return relu_grad

class MLP():
    def __init__(self, input_size, output_size, layers_size):
        self.weights_1 = np.random.randn(layers_size, input_size)
        self.weights_2 = np.random.randn(output_size, layers_size)
        self.bias_1 = np.random.randn(layers_size, 1)
        self.bias_2 = np.random.randn(output_size, 1)
        self.output_size = output_size

    def forward(self, x):
        self.u1 = []
        self.a1 = []
        self.u2 = []
        self.a2 = []

        for idx, w in enumerate(self.weights_1):
            self.u1.append(np.dot(w, x) + self.bias_1[idx][0])
        for u in self.u1:
            self.a1.append(relu(u))
        for idx, w in enumerate(self.weights_2):
            self.u2.append(np.dot(w, self.a1) + self.bias_2[idx][0])
        for u in self.u2:
            self.a2.append(sigmoid(u))
        return self.a2

    def backwards(self, y, x):
        """
        Last layer weight change
        """
        delta_c = self.derivative_loss(y, self.a2[0])
        delta_u2 = sigmoid_derivative(self.u2[0])
        delta = delta_c * delta_u2

        delta_w2 = np.multiply(delta, self.a1)
        delta_b2 = delta * 1


        """
        hidden layer weight change
        """
        delta_w1 = []
        delta_b1 = []

        delta_a1 = np.multiply(delta, self.weights_2[0])
        new_delta = delta_a1 * 1.0

        for idx, z in enumerate(self.u1):
            delta_u1 = derivative_relu(z)

            delta_a1 = delta * self.weights_2[0][idx]

            delta_ = delta_a1 * delta_u1

            delta_w1.append(np.array([delta_ * x[0], delta_ * x[1]]))
            delta_b1.append(delta_)

        delta_w1 = np.array(delta_w1)
        delta_b1 = np.array(delta_b1)

        return  -delta_w2, -delta_b2, -delta_w1, -delta_b1


    def loss(self, y, predict):
        return (predict - y)**2

    def derivative_loss(self, y, predict):
        return 2 * (predict - y)
