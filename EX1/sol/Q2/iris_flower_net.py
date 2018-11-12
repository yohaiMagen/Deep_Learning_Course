import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import numpy as np

def acuracy_rate(model, X, y):
    y_pred = model(X)
    maxes = torch.argmax(y_pred, 1)
    sum = torch.sum(torch.eq(maxes, y)).type(dtype)
    return (sum/maxes.shape[0]*100)

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
with open('iris.data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) is 0:
            continue
        train_labels.append(switcher[row[-1]])
        train_input.append([float(x) for x in row[:-1]])

dtype = torch.FloatTensor
state_matrix = torch.FloatTensor(train_input).type(dtype)
label_matrix = torch.Tensor(train_labels).type(torch.LongTensor)

input_dim = 4
output_dim = 3
h = 60
model = Net(input_dim, output_dim, h)

# cross_entropy = nn.CrossEntropyLoss().cuda()
cross_entropy = F.cross_entropy

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# train model
loss_list = []

iteration_number = 10000
loss_vec = []
for iteration in range(iteration_number):
    avg_lost = 0
    # forward pass
    y_pred = model(state_matrix)
    # Calculate Loss
    loss = cross_entropy(y_pred, label_matrix)
    # optimization
    optimizer.zero_grad()
    # backward propagation
    loss.backward()
    # Updating parameters
    optimizer.step()
    loss_vec.append(loss.data[0])

# now predict our Xor table
plt.plot(np.arange(iteration_number), loss_vec)
plt.title('iris classefir')
plt.xlabel('epoch')
plt.ylabel('Train loss')
plt.savefig('./Q2.png')
print(acuracy_rate(model, state_matrix, label_matrix))
plt.show()
