import torch.nn
from sklearn.model_selection import train_test_split
import numpy as np
import load_data

# input 2, output 1 hidden 30

epoch = 10


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

    def backward(self):
        self.torch.backward()
        return self

accuracies = []

data, labels = load_data.load_data()
# print(type(data))
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

criterion = torch.nn.MSELoss()

for iter in range(0,10): # run it 10 times

    model = MLP(2, 30)
    mySGD = torch.optim.SGD(model.parameters(), lr=load_data.learning_rate)
    epoch = 10

    """"
    model evaluation mode
    """
    model.eval()
    print(iter, "before:", criterion(y_test, model(x_test).squeeze()).item())

    """"
    model training mode
    """
    model.train()
    for e in range(0, epoch):
        mySGD.zero_grad() # gradient to 0
        loss = 0

        for idx, x in enumerate(x_train):
            y_pred = model(x)
            loss = criterion(y_train, y_pred.squeeze())
            loss.backward()
            mySGD.step()

    """"model evaluation mode"""

    print(iter, "after:", criterion(y_test, model(x_test).squeeze()).item())