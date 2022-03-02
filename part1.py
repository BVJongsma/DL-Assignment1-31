import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import numpy as np
import os


# load the dataset
def load_data():
    N = 500
    gq= sklearn.datasets.make_gaussian_quantiles(
        mean=None,
        cov=0.7,
        n_samples=N,
        n_features=2,
        n_classes=2,
        shuffle=True,
        random_state=33 #should be None
    )
    return gq


def forward():

    return 0


def backward():

    return 0


# def sigmoid(z):     return 1./(1.+np.exp(-z))


def dSigmoid(z):
    return sigmoid(z) *(1-sigmoid (z))


def load_MLP(data):
    myMLP = 5
    X = data[0]
    y = data[1]

    # initialise weights & biases


    # forward pass
    f = forward()

    # backward pass
    b = backward()

    # layers

    # stochastic gradient descent

    return f,b


#define activation functions for forward and backward pass

if __name__ == "__main__":
    gq = load_data()
    gqx = gq[0]
    gqy = gq[1]
    print(gqx.shape, gqy.shape)
    print(gqy)
    plt.title("Data distribution for 2-dimensional N(0,0.7) with 500 samples")
    plt.scatter(gqx[:, 0], gqx[:, 1], marker="o", c=gqy)
    #plt.show()

    f, b = load_MLP(gq)
