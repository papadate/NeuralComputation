import numpy as np
import math
import matplotlib.pyplot as plt

N = 400 # number of points per class
D = 2 # dimensionality
K = 4 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class class_data(nn.Module):
    def __init__(self):
        super(class_data, self).__init__()

        self.lin1 = torch.nn.Linear(2, 100)
        self.lin2 = torch.nn.Linear(100, 100)
        self.lin3 = torch.nn.Linear(100, 4)

    def forward(self, x):

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)

        return x

    # Prepare Training Data
    # X1 denotes the training examples
    # labels denote the groud truth with repsect to X1


X1 = torch.Tensor(X)
data_size = X1.shape[0]
X1 = X1.unsqueeze(0)
labels = y
labels = torch.Tensor(labels)
labels = labels.unsqueeze(0)

# Choose a proper classification function
# As we have 4 classes here, CrossEntropyloss will suit for this problem.
loss_fn = nn.CrossEntropyLoss()
classifier = class_data()
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
batchsize = 1600

for i in range(1000):
    choice = np.random.choice(data_size, batchsize, replace=False)
    X_ = X1[0, choice, :]
    labels_ = labels[0, choice]
    output = classifier(X_)
    loss = loss_fn(output, labels_.long())
    optimizer.zero_grad()
    loss.backward()
    if i % 100 == 0:
        print(loss)
    optimizer.step()

print('Training done')
preds = output.max(1)[1]

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

## ground truth show
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

## prediction show
fig = plt.figure()

plt.scatter(X[choice, 0], X[choice, 1], c=preds.numpy(), s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())



## The accuracy

predicted_results = preds.numpy()
ground_truth = y[choice]

acc = np.mean(predicted_results==ground_truth)

print ('training accuracy: ', acc)

#concatenating the meshgrid
xx = np.expand_dims(xx.flatten(), axis = 1)
yy = np.expand_dims(yy.flatten(), axis = 1)
val = np.concatenate((xx,yy),axis = 1)

#Converting the input to a tensor which the network will recognise
val = torch.from_numpy(val)
val = val.to(torch.float32)

#Calculating the label at each point of the meshgrid
probs = classifier(val)
seg = torch.argmax(probs,dim=1)

#Visualising the results
plt.figure()
plt.scatter(xx, yy, c=seg.numpy(), s=40, cmap=plt.cm.Spectral)
plt.figure()
plt.scatter(xx, yy, c=seg.numpy(), s=40, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
plt.show()