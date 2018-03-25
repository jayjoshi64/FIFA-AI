"""
The example of Breast cancer  mentioned in scikit-learn-code.py using Pytorch.
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sys

def create_nn(learning_rate=0.01, epochs=20):

    cancer = load_breast_cancer()

    print("loaded the data...")

    X = cancer['data']
    y = cancer['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    print(X_train.shape)

    X_train = Variable(torch.from_numpy(X_train)).float()
    X_test = Variable(torch.from_numpy(X_test)).float()
    
    y_train = Variable(torch.from_numpy(y_train)).float()
    y_test = Variable(torch.from_numpy(y_test)).float()

    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(30, 40)
            self.fc2 = nn.Linear(40, 20)
            self.fc3 = nn.Linear(20, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.sigmoid(x)

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.L1Loss()

    # run the main training loop
    for epoch in range(epochs):
        for nbt in range(3000):
            data, target = X_train, y_train
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            # data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if nbt % 1000 == 0:
                print('Train Epoch: {}  Loss: {:.6f}'.format(
                    epoch,  loss.data[0]))

    #test
    data, target = X_test, y_test

    net_out = net(data)
    net_out = net_out.data.numpy()
    y_pred = (net_out > 0.5)

    print("\n\n====== Confusion Matrix =======\n")
    print(confusion_matrix(y_test.data.numpy(),y_pred))

    print("\n\n=======Classification Report==========\n")
    print(classification_report(y_test.data.numpy(),y_pred))



if __name__ == "__main__":
    create_nn()