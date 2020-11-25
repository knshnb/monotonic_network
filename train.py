import numpy as np
import torch
import torch.nn as nn
from model import MonotonicNetwork
import matplotlib.pyplot as plt


def fit(model, x_train, y_train):
    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 1000
    inputs_all = torch.from_numpy(x_train)
    targets_all = torch.from_numpy(y_train)
    batches = np.split(np.arange(0, len(x_train)), 10)
    for epoch in range(num_epochs):
        for batch in batches:
            inputs = inputs_all[batch]
            targets = targets_all[batch]
            optimizer.zero_grad()
            outputs = model.inv(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
    return model


if __name__ == '__main__':
    width = 2.0
    num_data = 100
    net = torch.nn
    net = MonotonicNetwork(20, 20, const_sign=-1.0)
    X = np.random.uniform(-width, width, num_data).astype(np.float32)[:, None]
    eps = np.random.normal(0, 0.01, num_data).astype(np.float32)[:, None]
    Y = -X**3 + eps
    model = fit(net, X, Y)

    plt.plot(X, Y, 'ro', label='Original data')
    testX = np.linspace(-width, width, 50).astype(np.float32)[:, None]
    predicted = model(torch.from_numpy(testX)).detach().numpy()
    plt.plot(testX, predicted, label='Function curve')
    testX = np.linspace(-width, width, 50).astype(np.float32)[:, None]
    plt.plot(testX, testX, label='Y = X')
    testY = np.linspace(-width, width, 50).astype(np.float32)[:, None]
    predict2 = model.inv(torch.from_numpy(testY)).detach().numpy()
    plt.plot(testY, predict2, label='Inverse curve')
    plt.legend()
    plt.show()
