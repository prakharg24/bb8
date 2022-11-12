import torch
from tqdm import tqdm
import numpy as np

def train_epoch(model, train_loader, optimizer, criterion, cuda=True):
    model.train()
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()

        optimizer.step()

    return model

def get_predictions(model, valid_loader, cuda=True):
    model.eval()
    pred_arr = []
    label_arr = []

    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs.float())
            out_logits = torch.nn.functional.softmax(outputs.data, 1)
            _, predicted = torch.max(outputs.data, 1)

            pred_arr.extend(predicted.cpu().detach().numpy())
            label_arr.extend(labels.cpu().detach().numpy())

    return np.array(pred_arr), np.array(label_arr)
