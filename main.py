import argparse
import json
import os
from sklearn.metrics import accuracy_score

import torch
from torchvision import models

from dataloader import load_dataset
from train_utils import train_epoch, get_predictions
from utils import disparity_score, getScore, create_submission, load_state_dict
from dataorder import reorder_equal

CATEGORIES = ["skin_tone", "gender", "age"]
NUM_CLASSES_DICT = {"skin_tone": 10, "gender": 2, "age": 4}

def main(args):

    results = {'accuracy': {}, 'disparity': {}}
    for cat in CATEGORIES:
        train_loader, valid_loader, test_loader = load_dataset(cat=cat, batch_size=args.batch_size, img_size=args.img_size)

        num_classes = NUM_CLASSES_DICT[cat]
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, num_classes)
        load_state_dict(model, 'pretrained/resnet50_ft_weight.pkl')
        # print(model)
        # exit()
        # Finetune only last layer?? # Pretrain on faces?
        if args.cuda:
            model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        best_acc = -1
        for epoch in range(args.epochs):
            print("Epoch ", epoch)

            model = train_epoch(model, train_loader, optimizer, criterion, cuda=args.cuda)

            pred_arr, label_arr = get_predictions(model, valid_loader, cuda=args.cuda)

            acc = accuracy_score(label_arr, pred_arr)
            print("Accuracy : ", acc)

            scheduler.step()

            # if acc > best_acc:
            #     best_acc = acc
            #     torch.save(model, args.savefldr +'best.pth')

        # model = torch.load(args.savefldr +'best.pth')

        train_loader = reorder_equal(train_loader)
        model = train_epoch(model, train_loader, optimizer, criterion, cuda=args.cuda)
        # for epoch in range(10):
        #     train_loader_local = reorder_equal(train_loader)
        #     model = train_epoch(model, train_loader_local, optimizer, criterion, cuda=args.cuda)

        # Train a new model after hyperparameter tuning just for predictions
        pred_arr, label_arr = get_predictions(model, test_loader, cuda=args.cuda)
        results['accuracy'][cat] = accuracy_score(label_arr, pred_arr)
        results['disparity'][cat] = disparity_score(label_arr, pred_arr)

        print(results)

    print("Result Dictionary")
    print(results)
    create_submission(results, 'Baseline', 'baseline_score.json')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--img_size", type=int, default=64, help="Image Size (Height/Width)")
parser.add_argument("--epochs", type=int, default=5, help="Number of Training Epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Training")
parser.add_argument("--gpus", default="7", help="GPU Device ID to use")
parser.add_argument("--savefldr", default="default/", help="Folder to save checkpoints")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
main(args)
