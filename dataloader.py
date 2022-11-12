import numpy as np
import pandas as pd
import json
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

class BBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_dataset(cat='skin_tone', valid_split=0.2, batch_size=64, img_size=64):
    assert cat in ['skin_tone','gender','age']

    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ############################ Load Train Data ############################
    df = pd.read_csv('data/train/labels.csv')
    df_labeled = df[df[cat].notna()] # take only labeled data

    img_list = ['data/train/' + df_labeled.iloc[i]['name'] for i in range(df_labeled.shape[0])]
    labels = LabelEncoder().fit_transform(df_labeled[cat])

    img_list, img_list_valid, labels, labels_valid = train_test_split(img_list, labels, test_size=valid_split, random_state=0)
    img_list, labels = shuffle(img_list, labels, random_state=0)

    train_dataset = BBDataset(img_list, labels, preprocess)
    valid_dataset = BBDataset(img_list_valid, labels_valid, preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    ############################ Load Test Data ############################
    df_labeled = pd.read_csv('data/test/labels.csv')

    img_list = ['data/test/' + df_labeled.iloc[i]['name'] for i in range(df_labeled.shape[0])]
    labels = LabelEncoder().fit_transform(df_labeled[cat])

    test_dataset = BBDataset(img_list, labels, preprocess)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
