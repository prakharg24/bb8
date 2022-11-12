import argparse
import json
import os
from sklearn.metrics import accuracy_score

import torch
from torchvision import models

from dataloader import load_dataset
from train_utils import train_epoch, get_predictions
from utils import disparity_score, getScore, create_submission

CATEGORIES = ["skin_tone", "gender", "age"]

def main(args):

    results = {'accuracy': {}, 'disparity': {}}
    for cat in CATEGORIES:
        train_loader, valid_loader, test_loader = load_dataset(cat=cat, batch_size=args.batch_size, img_size=args.img_size)

        model = models.resnet50(pretrained=True)
        # Finetune only last layer?? # Pretrain on faces?
        if args.cuda:
            model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        for epoch in range(args.epochs):
            print("Epoch ", epoch)

            model = train_epoch(model, train_loader, optimizer, criterion, cuda=args.cuda)

            pred_arr, label_arr = get_predictions(model, valid_loader, cuda=args.cuda)

            acc = accuracy_score(label_arr, pred_arr)
            print("Accuracy : ", acc)

            ### SAVE MODEL

        pred_arr, label_arr = get_predictions(model, test_loader, cuda=args.cuda)
        results['accuracy'][cat] = accuracy_score(label_arr, pred_arr)
        results['disparity'][cat] = disparity_score(label_arr, pred_arr)

    print("Result Dictionary")
    print(results)
    create_submission(results, 'Baseline', 'baseline_score.json')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--img_size", type=int, default=64, help="Image Size (Height/Width)")
parser.add_argument("--epochs", type=int, default=5, help="Number of Training Epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Training")
parser.add_argument("--gpus", default="7", help="GPU Device ID to use")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
main(args)
