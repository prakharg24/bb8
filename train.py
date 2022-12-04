import argparse
import json
import os
import math
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torchvision import models

from dataloader import load_dataset
from train_utils import train_epoch, get_predictions
from utils import disparity_score, getScore, create_submission, load_state_dict, randomness_score

CATEGORIES = ["skin_tone", "gender", "age"]
NUM_CLASSES_DICT = {"skin_tone": 10, "gender": 2, "age": 4}
CHI_MULT_DICT = {"skin_tone": 1.3, "gender": 1.1, "age": 1.2}

def train(args):

    results = {'accuracy': {}, 'disparity': {}}
    for cat in CATEGORIES:
        print("Training for Category : ", cat)
        train_loader, valid_loader, test_loader = load_dataset(cat=cat, valid_split=0., batch_size=args.batch_size)

        if valid_loader is None:
            valid_loader = test_loader

        num_classes = NUM_CLASSES_DICT[cat]
        model = models.resnet50(weights='DEFAULT')
        model.fc = torch.nn.Linear(2048, num_classes + 1)
        load_state_dict(model, 'pretrained/resnet50_ft_weight.pkl')

        model = torch.nn.DataParallel(model)

        if args.cuda:
            model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader)*args.epochs//2)

        best_score = -1
        for epoch in range(args.epochs):
            print("Epoch ", epoch)
            print("Learning Rate ", scheduler.get_last_lr())

            model = train_epoch(model, train_loader, optimizer, criterion, scheduler, cuda=args.cuda)

            # if valid_loader is not None:
                # pred_arr, label_arr, bg_arr = get_predictions(model, valid_loader, cuda=args.cuda, bg_class=num_classes)
            pred_arr, label_arr, bg_arr = get_predictions(model, train_loader, cuda=args.cuda, bg_class=num_classes)
            acc, disp, chi_sq_bool = accuracy_score(label_arr, pred_arr), disparity_score(label_arr, pred_arr), randomness_score(pred_arr[bg_arr], num_classes)
            score = num_classes*acc*(1 - disp**(num_classes/2))
            if chi_sq_bool:
                score = score * CHI_MULT_DICT[cat]
            print("Accuracy, Disparity, Chi Square Test, Score : ", acc, disp, chi_sq_bool, score)
            if score > best_score:
                best_score = score
                torch.save(model, args.savefldr + cat + '_best.pth')

            torch.save(model, args.savefldr + cat + '_last.pth')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--epochs", type=int, default=5, help="Number of Training Epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Training")
parser.add_argument("--gpus", default=None, help="GPU Device ID to use")
parser.add_argument("--savefldr", default="models/", help="Folder to save checkpoints")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
train(args)
