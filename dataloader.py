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

        return image.cuda(), label

def load_dataset(cat='skin_tone', valid_split=0.2, batch_size=64):
    assert cat in ['skin_tone','gender','age']

    preprocess_train = transforms.Compose([
        transforms.RandomResizedCrop(416),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_test = transforms.Compose([
        transforms.Resize(452),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ############################ Load Train Data ############################
    df = pd.read_csv('data/train/labels_face.csv')
    df_labeled = df[df[cat].notna()] # take only labeled data

    img_list = ['data/train/' + df_labeled.iloc[i]['name'] for i in range(df_labeled.shape[0])]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_labeled[cat])

    is_not_face = np.logical_not(df_labeled['is_face'])
    labels[is_not_face] = np.max(labels) + 1

    # print("Classes and Unique Counts")
    # print(np.unique(labels, return_counts=True))

    # if valid_split > 0.:
    #     img_list, img_list_valid, labels, labels_valid = train_test_split(img_list, labels, test_size=valid_split, random_state=0)
    #     valid_dataset = BBDataset(img_list_valid, labels_valid, preprocess_test)
    #     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    # else:
    #     valid_loader = None

    img_list, labels = shuffle(img_list, labels, random_state=0)
    train_dataset = BBDataset(img_list, labels, preprocess_train)

    ############################ Load Test Data ############################
    df_labeled = pd.read_csv('data/test/labels_face.csv')

    img_list = ['data/test/' + df_labeled.iloc[i]['name'] for i in range(df_labeled.shape[0])]
    labels = label_encoder.transform(df_labeled[cat])
    is_not_face = np.logical_not(df_labeled['is_face'])
    labels[is_not_face] = np.max(labels) + 1

    test_dataset = BBDataset(img_list, labels, preprocess_test)

    comp_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(comp_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # return train_loader, valid_loader, test_loader
    return train_loader, None, None


def load_dataset_inference(cat='skin_tone', batch_size=64, label_file='data/test/labels.csv', img_parent_dir='data/test/'):
    assert cat in ['skin_tone','gender','age']

    preprocess_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ############################ Load Test Data ############################
    df_labeled = pd.read_csv(label_file)

    img_list = [img_parent_dir + df_labeled.iloc[i]['name'] for i in range(df_labeled.shape[0])]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_labeled[cat])

    test_dataset = BBDataset(img_list, labels, preprocess_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
