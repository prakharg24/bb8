import argparse
import json
import os
import math
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import pytorch_warmup as warmup
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import models

from dataloader import load_dataset
from train_utils import train_epoch, get_predictions
from utils import disparity_score, getScore, create_submission, load_state_dict, randomness_score
from dataorder import reorder_equal
from senet import senet50

CATEGORIES = ["skin_tone", "gender", "age"]
# CATEGORIES = ["skin_tone"]
NUM_CLASSES_DICT = {"skin_tone": 10, "gender": 2, "age": 4}
CHI_MULT_DICT = {"skin_tone": 1.3, "gender": 1.1, "age": 1.2}
# CLASS_WEIGHTS = {"skin_tone": [350,   81,  882, 1180, 1106,  903,  434,  476,  332,  199, 2610],
#                  "gender": [3551, 2392, 2610],
#                  "age": [1306, 2969, 1481,  187, 2610]}

# for ele in CLASS_WEIGHTS:
#     arr_max = np.max(CLASS_WEIGHTS[ele])
#     CLASS_WEIGHTS[ele] = [math.sqrt(arr_max/e) for e in CLASS_WEIGHTS[ele]]

def train(args):

    results = {'accuracy': {}, 'disparity': {}}
    for cat in CATEGORIES:
        train_loader, valid_loader, test_loader = load_dataset(cat=cat, valid_split=0., batch_size=args.batch_size)

        if valid_loader is None:
            valid_loader = test_loader

        num_classes = NUM_CLASSES_DICT[cat]
        model = models.resnet50(weights='DEFAULT')
        # model = models.resnet152(weights='DEFAULT')
        # model = senet50()
        model.fc = torch.nn.Linear(2048, num_classes + 1)
        load_state_dict(model, 'pretrained/resnet50_ft_weight.pkl')
        # load_state_dict(model, 'pretrained/senet50_ft_weight.pkl')

        # Finetune only last layer?? # Pretrain on faces?
        if args.cuda:
            model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(CLASS_WEIGHTS[cat]).cuda())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader)*args.epochs//4)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=len(train_loader))
        # skip_scheduler = False
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # swa_model = AveragedModel(model)
        # swa_start_epoch = int(0.9*args.epochs)
        best_score = -1
        for epoch in range(args.epochs):
            print("Epoch ", epoch)
            print("Learning Rate ", scheduler.get_last_lr())
            # if epoch >= swa_start_epoch:
            #     skip_scheduler = True

            model = train_epoch(model, train_loader, optimizer, criterion, scheduler, warmup_scheduler, cuda=args.cuda)
            # model = train_epoch(model, train_loader, optimizer, criterion, scheduler, warmup_scheduler, cuda=args.cuda, skip_scheduler=skip_scheduler)
            # if skip_scheduler:
            #     swa_model.update_parameters(model)

            if valid_loader is not None:
                pred_arr, label_arr, bg_arr = get_predictions(model, valid_loader, cuda=args.cuda, bg_class=num_classes)
                acc, disp, chi_sq_bool = accuracy_score(label_arr, pred_arr), disparity_score(label_arr, pred_arr), randomness_score(pred_arr[bg_arr], num_classes)
                score = num_classes*acc*(1 - disp**(num_classes/2))
                if chi_sq_bool:
                    score = score * CHI_MULT_DICT[cat]
                print("Accuracy, Disparity, Chi Square Test, Score : ", acc, disp, chi_sq_bool, score)
                if score > best_score:
                    best_score = score
                    torch.save(model, args.savefldr + cat + '_best.pth')

        # torch.optim.swa_utils.update_bn(train_loader, swa_model)
        # torch.save(swa_model.module, args.savefldr + cat + '_best.pth')

        # if valid_loader is not None:
        #     pred_arr, label_arr, bg_arr = get_predictions(swa_model.module, valid_loader, cuda=args.cuda, bg_class=num_classes)
        #     acc, disp, chi_sq_bool = accuracy_score(label_arr, pred_arr), disparity_score(label_arr, pred_arr), randomness_score(pred_arr[bg_arr], num_classes)
        #     score = num_classes*acc*(1 - disp**(num_classes/2))
        #     if chi_sq_bool:
        #         score = score * CHI_MULT_DICT[cat]
        #     print("Accuracy, Disparity, Chi Square Test, Score : ", acc, disp, chi_sq_bool, score)


        # for epoch in range(args.epochs):
        #     print("FairOrder Epoch ", epoch)
        #
        #     train_loader_local = reorder_equal(train_loader)
        #     model = train_epoch(model, train_loader_local, optimizer, criterion, cuda=args.cuda)
        #
        #     if valid_loader is not None:
        #         pred_arr, label_arr = get_predictions(model, valid_loader, cuda=args.cuda)
        #         acc, disp = accuracy_score(label_arr, pred_arr), disparity_score(label_arr, pred_arr)
        #         score = num_classes*acc*(1 - disp**(num_classes/2))
        #         print("Accuracy, Disparity, Score : ", acc, disp, score)
        #         if score > best_score:
        #             best_score = score
        #             torch.save(model, args.savefldr +'best.pth')
        #
        #     # scheduler.step()
        #
        # model = torch.load(args.savefldr + 'best.pth')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--epochs", type=int, default=5, help="Number of Training Epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Training")
parser.add_argument("--gpus", default="7", help="GPU Device ID to use")
parser.add_argument("--savefldr", default="models/", help="Folder to save checkpoints")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
train(args)
