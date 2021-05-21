"""

QUESTION 1

Some helpful code for getting started.


"""

import numpy as np
import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from imagenet10 import ImageNet10

import pandas as pd
import os
import matplotlib.pyplot as plt
import model
from config import *
from model import AlexNet






# Gathers the meta data for the images

paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)
            
data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data

# See what the dataframe now contains
#print("Found", len(data_df), "images.")
# If you want to see the image meta data
#print(data_df.head())



# Split the data into train and test sets and instantiate our new ImageNet10 objects.
train_split = 0.80 # Defines the ratio of train/valid data.

# valid_size = 1.0 - train_size
train_size = int(len(data_df)*train_split)


data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)


# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

# See what you've loaded
# print("len(dataset_train)", len(dataset_train))
# print("len(dataset_valid)", len(dataset_valid))
#
# print("len(train_loader)", len(train_loader))
# print("len(valid_loader)", len(valid_loader))


# def timshow(x):
#     xa = np.transpose(x.numpy(),(1,2,0))
#     plt.imshow(xa)
#     plt.show()


# Computation of loss and accuracy for given dataset loader and model
def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            running_loss += loss
            n += 1
    return running_loss/n, correct/total



# Define model
# net = AlexNet(num_classes=10, init_weights=True)
# net = nn.Sequential(
#     nn.Conv2d(3, 16, kernel_size=5, padding=0),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(16, 32, kernel_size=5, padding=0),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(32, 64, kernel_size=5, padding=0),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(64*12*12, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10)
# )
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 32, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 64, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv3 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 64, 3, 1, 1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.dense = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(64 * 16 * 16, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 10)
#         )
#
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(conv1_out)
#         conv3_out = self.conv3(conv2_out)
#         res = conv3_out.view(conv3_out.size(0), -1)
#         out = self.dense(res)
#         return out


net = AlexNet()




if __name__ == '__main__':
    # get some random training images using the data loader
    dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    inputs, labels = dataiter.next()
    # show images and labels
    # timshow(torchvision.utils.make_grid(images))
    #
    # print(f"labels {[CLASS_LABELS[i] for i in range(10)]}")

    # Train model
    nepochs = 40
    statsrec = np.zeros((3, nepochs))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(nepochs):  # loop over the dataset multiple times

        running_loss = 0.0
        n = 0
        # for i, data in enumerate(train_loader, 0):
        # inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # accumulate loss
        running_loss += loss.item()
        # n += 1

        ltrn = running_loss
        ltst, atst = stats(valid_loader, net)
        statsrec[:,epoch] = (ltrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")

    fig, ax1 = plt.subplots()
    plt.plot(statsrec[0], 'r', label = 'training loss', )
    plt.plot(statsrec[1], 'g', label = 'test loss' )
    plt.legend(loc='center')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.title('Training and test loss, and test accuracy')
    plt.title('Training and test loss')
    # ax2=ax1.twinx()
    # ax2.plot(statsrec[2], 'b', label = 'test accuracy')
    # ax2.set_ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.show()


