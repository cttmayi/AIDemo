import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Synthetic dataset
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
from collections import Counter
# Visualization
import matplotlib.pyplot as plt
# Model and performance
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


device = 'cpu'# 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using device: {device}')

pth = 'model.pth'
n_status = 10
n_action = 10

def make_dataset(n_status, n_action):

    # Create an imbalanced dataset
    X, y = make_classification(n_samples=10000, n_features=n_status, n_informative=10,
                            n_redundant=0, n_repeated=0, n_classes=n_action,
                            n_clusters_per_class=1,
                            # weights=[0.95, 0.05],
                            class_sep=0.5, random_state=0)

    # onehot encode the labels(numpy array)
    y = np.eye(n_action)[y]

    print('X = ', X[0].tolist())
    print('y = ', y[0].tolist())

    # 将X的偶数行和奇数行连接起来，形成新的X
    # X = np.concatenate((X[::2], X[1::2]), axis=1)
    # y = np.concatenate((y[::2], y[1::2]), axis=0)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, y_train, X_test, y_test


class Net(nn.Module):
    def __init__(self, n_status, n_action):
        super(Net, self).__init__()

        hidden_dim = 256
        self.status = nn.Sequential(
            nn.Linear(n_status, hidden_dim*10),
            nn.ReLU(),
            nn.Linear(hidden_dim*10, hidden_dim),
            nn.ReLU(),
        )
        self.action = nn.Sequential(
            nn.Linear(hidden_dim, n_action),
            nn.Softmax(dim=1)  # Assuming action_dim is the number of classes
        )
    
    def forward(self, x):
        status = self.status(x)
        action = self.action(status)
        return action


def load_model(n_status, n_action, pth):
    net = Net(n_status=n_status, n_action=n_action).to(device)
    # Load the model if it exists
    try:
        net.load_state_dict(torch.load(pth))
        print(f'Model loaded from {pth}')
    except FileNotFoundError:
        print(f'No model found at {pth}, starting from scratch.')
    except Exception as e:
        print(f'Error loading model: {e}, starting from scratch.')

    return net


def train_model(net, time_steps, X, y, criterion, optimizer):
    # Train the model
    for epoch in range(time_steps):
        # Forward pass
        outputs = net(X)
        loss = criterion(outputs, y)
        # loss = -torch.mean(torch.sum(y_train*outputs, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, time_steps, loss.item()))


def test_model(net, X, y, threshold=0.5):
    with torch.no_grad():
        outputs = net(X)
        predicted = torch.argmax(outputs.data, dim=1)
        y_labels = torch.argmax(y, dim=1)
        # Calculate accuracy
        accuracy = torch.sum(predicted == y_labels).item() / y_labels.size(0)
        print('Test Accuracy 1: {:.2f}%'.format(100 * accuracy))

        accuracy = torch.sum(torch.gather(outputs, 1, y_labels.unsqueeze(1))>threshold).item() / y_labels.size(0)
        print('Test Accuracy 2: {:.2f}%'.format(100 * accuracy))
        
        # print('Predicted labels:', predicted.numpy()[0:5])
        # print('True labels:', y_labels.numpy()[0:5])
        # print('True labels one-hot:', yy_test.numpy()[0:5])
        # print('Predicted labels one-hot:', outputs.numpy()[0])


X_train, y_train, X_test, y_test = make_dataset(n_status, n_action)

net = load_model(n_status, n_action, pth)

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Train the model
train_model(net, time_steps=1000, X=X_train, y=y_train, criterion=criterion, optimizer=optimizer)

# Test the model
test_model(net, X_train, y_train, threshold=0.5)
test_model(net, X_test, y_test, threshold=0.001)

# Save the model
torch.save(net.state_dict(), pth)
