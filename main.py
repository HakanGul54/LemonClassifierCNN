from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 1, bias=False),
            nn.MaxPool2d(5, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, 2, 1, bias=False),
            nn.MaxPool2d(5, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, data_loader, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_root = "Lemon"

    data = dset.ImageFolder(root=data_root,
                            transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    data_loader = torch.utils.data.DataLoader(data, batch_size=5, shuffle=True, num_workers=2)

    data_test = dset.ImageFolder(root="LemonTest",
                                transform=transforms.Compose([
                                    transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=5, shuffle=True, num_workers=2)

    class_names = data.classes

    # Get a batch of training data
    inputs, classes = next(iter(data_loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    plt.show()

    model = ClassificationModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    num_epochs = 15
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        loss_data = 0
        acc_data = 0

        for inputs, labels in data_loader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_data += loss.item() * inputs.size(0)
            acc_data += torch.sum(predictions == labels.data)

        epoch_loss = loss_data / len(data)
        epoch_acc = acc_data.double() / len(data)
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    visualize_model(model, data_loader_test, 10)
    plt.show()
