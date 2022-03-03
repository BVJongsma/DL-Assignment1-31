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
    def __init__(self, input_size, layers_size, output_size, batch_size):
        self.weights_1 = np.random.uniform(-1/math.sqrt(input_size), 1/math.sqrt(input_size), (layers_size, input_size))
        self.weights_2 = np.random.uniform(-1/math.sqrt(layers_size), 1/math.sqrt(layers_size), (output_size, layers_size))
        self.bias_1 = np.random.randn(layers_size, 1)
        self.bias_2 = np.random.randn(output_size, 1)
        self.output_size = output_size
        self.batch_size = batch_size
        self.layer_size = layers_size
        self.input_size = input_size

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

    def update_batch(self, y, x, lr):
        w1 = np.zeros((self.layer_size, self.input_size))
        w2 = np.zeros((self.output_size, self.layer_size))
        b1 = np.zeros((self.layer_size, 1))
        b2 = np.zeros((self.output_size, 1))

        for iter in range(self.batch_size):
            self.forward(x[iter])
            delta_w2, delta_b2, delta_w1, delta_b1 = self.backwards(y[iter], x[iter])

            w1 += delta_w1
            w2 += delta_w2
            b1 += delta_b1
            b2 += delta_b2

        self.weights_2 += (w2 / self.batch_size) * lr
        self.weights_1 += (w1 / self.batch_size) * lr
        self.bias_2 += (b2 / self.batch_size) * lr
        self.bias_1 += (b1 / self.batch_size) * lr

    def backwards(self, y, x):
        """
        Last layer weight change
        """
        delta_w2 = []
        delta_b2 = []
        delta = []

        for y_idx, a, u in zip(y, self.a2, self.u2):
            delta_c = self.derivative_loss(y_idx, a)
            delta_u2 = sigmoid_derivative(u)
            delta.append(delta_c * delta_u2)
            delta_w2.append(np.multiply(delta[-1], self.a1))
            delta_b2.append(delta[-1])

        delta_w2 = np.array(delta_w2)
        delta_b2 = np.array([delta_b2]).T
        delta = np.array(delta)

        """
        hidden layer weight change
        """
        delta_w1 = []
        delta_b1 = []

        for idx, z in enumerate(self.u1):
            delta_a1 = 0
            delta_u1 = derivative_relu(z)
            for id, d in enumerate(delta):
                delta_a1 += d * self.weights_2[id][idx]
            delta_ = delta_a1 * delta_u1
            delta_w1.append(delta_ * x)
            delta_b1.append(delta_)

        delta_w1 = np.array(delta_w1)
        delta_b1 = np.array([delta_b1]).T
        return  -delta_w2, -delta_b2, -delta_w1, -delta_b1


    def loss(self, y, predict):
        return (predict - y)**2

    def derivative_loss(self, y, predict):
        return 2 * (predict - y)
