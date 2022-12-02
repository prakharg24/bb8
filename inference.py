import argparse
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torchvision import models

from dataloader import load_dataset_inference
from train_utils import train_epoch, get_predictions
from utils import disparity_score, getScore, create_submission, load_state_dict, randomness_score
from dataorder import reorder_equal

CATEGORIES = ["skin_tone", "gender", "age"]
# CATEGORIES = ["skin_tone"]
NUM_CLASSES_DICT = {"skin_tone": 10, "gender": 2, "age": 4}
CHI_MULT_DICT = {"skin_tone": 1.3, "gender": 1.1, "age": 1.2}

def inference(args):

    results = {'accuracy': {}, 'disparity': {}}
    for cat in CATEGORIES:
        test_loader = load_dataset_inference(cat=cat, batch_size=args.batch_size)

        num_classes = NUM_CLASSES_DICT[cat]
        model = torch.load(args.savefldr + cat + '_best.pth')

        if args.cuda:
            model = model.cuda()

        pred_arr, label_arr = get_predictions(model, test_loader, cuda=args.cuda, bg_class=num_classes, inference_only=True)
        results['accuracy'][cat] = accuracy_score(label_arr, pred_arr)
        results['disparity'][cat] = disparity_score(label_arr, pred_arr)
        print(results)

    print("Result Dictionary")
    print(results)
    create_submission(results, 'Baseline', args.outfile)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
parser.add_argument("--gpus", default="7", help="GPU Device ID to use")
parser.add_argument("--savefldr", default="models/", help="Folder to save checkpoints")
parser.add_argument("--cuda", action="store_true", help="Use CUDA")
parser.add_argument("--outfile", default="baseline_score.json", help="Output file name")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
inference(args)
