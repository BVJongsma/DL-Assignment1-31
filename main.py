import sklearn.datasets
import load_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mlp


data, labels = load_data.load_data()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=1)
'''
"""
Pre training accuracy:
"""
accuracy = 0
for idx, x in enumerate(X_test):
    y_hat = network.forward(x)
    if y_hat[0] > 0.5:
        y_hat = 1
    else:
        y_hat = 0

    if y_hat == y_test[idx]:
        accuracy += 1
accuracy = accuracy / len(X_test)
print(accuracy)
'''
learning_rate = load_data.learning_rate
accuracies = []
for iter in range(0, 10):

    network = mlp.MLP(2, 1, 30)

    """
    training sequence
    """
    epochs = 10
    # training data

    for e in range(epochs):
        loss = 0
        for idx, x in enumerate(X_train):
            y_hat = network.forward(x)

            loss += network.loss(y_train[idx], y_hat[0])

            w2, b2, w1, b1 = network.backwards(y_train[idx], x)
            network.weights_2 += w2 * learning_rate
            network.bias_2 += b2 * learning_rate
            network.weights_1 += w1 * learning_rate
            network.bias_1 += np.expand_dims(b1,axis=1) * learning_rate

    """
    post training accuracy:
    """
    accuracy = 0
    for idx, x in enumerate(X_test):
        y_hat = network.forward(x)
        if y_hat[0] > 0.5:
            y_hat = 1
        else:
            y_hat = 0

        if y_hat == y_test[idx]:
            accuracy += 1
    accuracy = accuracy / len(y_test)
    print(accuracy)
    accuracies.append(accuracy)
print(accuracies)