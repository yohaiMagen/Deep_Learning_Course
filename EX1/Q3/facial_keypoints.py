import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eval(model, data, labels):
    predicted = model(data)
    errors = F.mse_loss(predicted, labels)
    return float(errors) / predicted.shape[0]

class Net(nn.Module):
    def __init__(self, input_dim, output_dimension, hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dimension)

    def forward(self, x):
        x = self.fc1(x)
        x = Net._requ(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def _requ(x):
        return F.relu(x).pow(2)


switcher = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

train_input = []
train_labels = []

# loading data
data_frame = pd.read_csv('training.csv', sep=',').dropna()
Y = (data_frame.iloc[:, 0:-1].values / 48) - 1

img_df = data_frame.iloc[:, -1]

img_df.to_csv('__img.csv', index=False)
X = pd.read_csv('__img.csv', sep=' ', header=None).values / 255

Y_t = ((Y + 1) * 48).reshape(-1, 2, 15)
plt.imshow(X[101].reshape(96, 96).T, cmap="gray")
plt.scatter(Y_t[101, 1, :], Y_t[101, 0, :])
plt.show()

mask = np.arange(0, 2140)
np.random.shuffle(mask)
val_mask = mask[:int(X.shape[0] * 0.15)]
train_mask = mask[int(X.shape[0] * 0.15):]
X_val = torch.FloatTensor(X[val_mask])
X_train = torch.FloatTensor(X[train_mask])
Y_val = torch.FloatTensor(Y[val_mask])
Y_train = torch.FloatTensor(Y[train_mask])


C = 1
L = 96
W = 96
N = 2140
flatten_dim = L * W
output_dim = 30
h = 100

linear_model = torch.nn.Sequential(nn.Linear(flatten_dim, h), nn.ReLU(), nn.Linear(h, output_dim))

# cross_entropy = nn.CrossEntropyLoss().cuda()
loss_fn = nn.MSELoss()

optimizer = optim.Adam(linear_model.parameters(), lr=1e-2)
# train model
train_loss = []
val_loss = []


iteration_number = 5
for iteration in range(iteration_number):
    # forward pass
    y_pred = linear_model(X_train)
    # Calculate Loss
    loss = loss_fn(y_pred, Y_train)
    # optimization
    optimizer.zero_grad()
    # backward propagation
    loss.backward()
    # Updating parameters
    optimizer.step()
    train_loss.append(eval(linear_model, X_train, Y_train))
    val_loss.append(eval(linear_model, X_val, Y_val))


plt.plot(np.arange(iteration_number), train_loss)
plt.plot(np.arange(iteration_number), val_loss)
pred = ((linear_model(X_val) + 1) * 48).detach().numpy().reshape(-1, 2, 15)
Y_val_t = ((Y_val + 1) * 48).numpy().reshape(-1, 2, 15)
plt.imshow(X_val[100].numpy().reshape(96, 96), cmap="gray")
plt.scatter(pred[100, 1, :], pred[100, 0, :])
plt.scatter(Y_val_t[100, 1, :], Y_val_t[100, 0, :])
plt.show()
# now predict our Xor table



