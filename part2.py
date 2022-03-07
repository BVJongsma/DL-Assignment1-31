import torch.nn
from sklearn.model_selection import train_test_split
import load_data
import matplotlib.pyplot as plt

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


accuracies = []

data, labels = load_data.load_data()
# print(type(data))
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
x_validate = torch.from_numpy(x_validate).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_validate = torch.from_numpy(y_validate).float()

criterion = torch.nn.MSELoss()


def accuracy(pred, true):
    acc = 0
    for y_hat, y in zip(pred, true):
        y_ = 0
        if y_hat.item() > 0.5:
            y_ = 1
        else:
            y_ = 0
        if y_ == y.item():
            acc += 1

    return acc / len(pred)


for iter in range(0, 10):  # run it 10 times

    model = MLP(2, 30)
    mySGD = torch.optim.SGD(model.parameters(), lr=0.1)
    epoch = 50

    """
    model evaluation mode
    """
    model.eval()
    print("Iteration:", iter, "before:", accuracy(model(x_test), y_test))

    """
    model training mode
    """
    loss_ = 0
    training_loss = []
    validation_loss = []

    for e in range(0, epoch):
        model.train()
        loss_ = 0
        for x, y in zip(x_train, y_train):
            mySGD.zero_grad()  # gradient to 0
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss_ += loss
            loss.backward()
            mySGD.step()
        training_loss.append((loss_/len(y_train)).item())

        loss_ = 0
        model.eval()
        for x, y in zip(x_validate, y_validate):
            y_pred = model(x)
            loss_ += criterion(y_pred.squeeze(), y)
        validation_loss.append((loss_/len(y_validate)).item())

    # training_loss = np.array(training_loss)
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    """
    model evaluation mode
    """
    model.eval()
    y_pred = model(x_test).squeeze()
    acc = accuracy(y_pred, y_test)
    accuracies.append(acc)
    print("Iteration:", iter, "after:", acc)

print(accuracies)