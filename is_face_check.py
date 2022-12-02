import sys
import torch
import numpy as np
import pandas as pd
from torchvision import models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from facenet_pytorch import MTCNN

parent_fldr = sys.argv[1]
df = pd.read_csv(parent_fldr + 'labels.csv')

model = MTCNN(device='cuda')

img_list = [parent_fldr + df.iloc[i]['name'] for i in range(df.shape[0])]
is_face_list = []

for filename in tqdm(img_list):
    input_image = Image.open(filename).convert('RGB')
    with torch.no_grad():
        boxes, probs, landmarks = model.detect(input_image, landmarks=True)

    if probs[0] == None or np.max(probs) < 0.9:
        is_face_list.append(False)
        # print("Is not face : ", filename)
    else:
        is_face_list.append(True)
        # print("Is face : ", filename)

df['is_face'] = is_face_list
df.to_csv(parent_fldr + 'labels_face.csv', index=False)
