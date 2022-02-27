import torch.nn
import part1
from sklearn.model_selection import train_test_split

# input 2, output 1 hidden 30

epochs = 10

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
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

    def backward(self,y,w):

        return self


model = MLP(2, 30)
loss = torch.nn.MSELoss()
data = part1.load_data()
trainx, testx, trainy, testy = train_test_split(data[0], data[1], test_size=0.2)



