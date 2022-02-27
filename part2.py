import torch.nn
import part1
from sklearn.model_selection import train_test_split
import numpy as np

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


model = MLP(2, 30)
loss = torch.nn.MSELoss()
mySGD = torch.optim.SGD(model.parameters(), lr=0.01)  # why does model.parameters() work?
data = part1.load_data()
print(type(data))
x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

model.eval()
print("accuracy before trained model:", loss(y_test, model(x_test)).item())

model.train()
for epoch in range(epoch):
    mySGD.zero_grad()
    predy = model(x_train)
    l = loss(y_train, predy)
    print("epoch ", epoch, l.item())
    l.backward()
    mySGD.step()

model.eval()
y_pred = model(x_test)

model.eval()
print("accuracy after trained model:", loss(y_test, model(x_test)).item())
