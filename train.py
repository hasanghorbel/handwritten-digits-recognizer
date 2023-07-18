import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from neuralnet import NeuralNet, hidden_size, input_size, num_classes

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper_parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST train dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape (100, 1, 28, 28)
        # output size (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(
                f'Epoch [{(epoch+1)}/{(num_epochs)}], step [{(i+1)}/{(n_total_steps)}], Loss {loss.item()}')

print('Finished training')

parser = argparse.ArgumentParser(description='recognize handwritten digits')
parser.add_argument('-p', '--path', default='model',
                    type=str, help='save directory path')
args = parser.parse_args()

PATH = args.path
if not os.path.exists(PATH):
    os.mkdir(PATH)
torch.save(model.state_dict(), os.path.join(PATH, 'model.pth'))
