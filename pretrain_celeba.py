import os
import glob
import numpy as np
from matplotlib import pyplot
from PIL import Image

import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms

class celeba(Dataset):
    def __init__(self, data_path=None, label_path=None):
        self.data_path = data_path
        self.label_path = label_path

        self.transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = torch.Tensor(self.label_path[idx])

        return image_tensor, image_label


def train(model, epochs, train_all_losses, train_all_acc):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        labels = torch.Tensor(labels)

        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        outputs = torch.sigmoid(model(inputs))
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        result = outputs > 0.5
        correct += (result == labels).sum().item()

        if i % 64 == 0:
            print('Training set: [Epoch: %d, Data: %6d] Loss: %.3f' %
                  (epochs + 1, i * 64, loss.item()))

    acc = correct / (split_train * 40)
    running_loss /= len(trainloader)
    train_all_losses.append(running_loss)
    train_all_acc.append(acc)
    print('\nTraining set: Epoch: %d, Accuracy: %.2f %%' % (epochs + 1, 100. * acc))

def validation(model, val_all_losses, val_all_acc, best_acc):
    model.eval()
    validation_loss = 0.0
    correct = 0
    for data, target in validloader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = torch.sigmoid(model(data))

        validation_loss += criterion(output, target).item()

        result = output > 0.5
        correct += (result == target).sum().item()


    validation_loss /= len(validloader)
    acc = correct / (len(validloader) * 40)

    val_all_losses.append(validation_loss)
    val_all_acc.append(acc)

    print('\nValidation set: Average loss: {:.3f}, Accuracy: {:.2f}%)\n'.format(validation_loss, 100. * acc))

    return acc


data_path = sorted(glob.glob('/mnt/LargeDisk/Data/celeba/img_align_celeba/img_align_celeba/*.jpg'))

label_path = "/mnt/LargeDisk/Data/celeba/list_attr_celeba.csv"
label_list = open(label_path).readlines()[1:]

data_label = []
for i in range(len(label_list)):
    data_label.append(label_list[i].strip().split(','))

for m in range(len(data_label)):
    data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
    data_label[m] = [int(p) for p in data_label[m]]

dataset = celeba(data_path, data_label)
indices = list(range(202599))
split_train = 162079
train_idx, valid_idx = indices[:split_train], indices[split_train:]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)
validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)

print(len(trainloader))
print(len(validloader))

# define empty list to store the losses and accuracy for ploting
train_all_losses = []
train_all_acc = []
val_all_losses = []
val_all_acc = []
# define the training epoches
epochs = 100

# instantiate Net class
model = models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, 40)
# use cuda to train the network
model.to('cuda')
#loss function and optimizer
criterion = nn.BCELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

best_acc = 0.0
for epoch in range(epochs):
    train(model, epoch, train_all_losses, train_all_acc)
    acc = validation(model, val_all_losses, val_all_acc, best_acc)
    if acc > best_acc:
        checkpoint_path = 'pretrained/celeba_resnet18.pth'
        best_acc = acc
        # save the model and optimizer
        torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print('new best model saved')
    print("========================================================================")
