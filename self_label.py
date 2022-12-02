import torch
import numpy as np
import pandas as pd
from torchvision import models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import math

from train_utils import get_predictions
from sklearn.preprocessing import LabelEncoder

CATEGORIES = ["skin_tone", "gender", "age"]
NUM_CLASSES_DICT = {"skin_tone": 10, "gender": 2, "age": 4}

parent_fldr = 'data/train/'
df = pd.read_csv(parent_fldr + 'labels_face.csv')
img_list = [parent_fldr + df.iloc[i]['name'] for i in range(df.shape[0])]

preprocess_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(df.head(10))
for cat in CATEGORIES:
    model = torch.load('models/augmix_warmup/' + cat + '_best.pth')

    df_labeled = df[df[cat].notna()]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_labeled[cat])

    outlist = []
    for i, filename in tqdm(enumerate(img_list)):

        if isinstance(df[cat].iloc[i], float) and math.isnan(df[cat].iloc[i]):
            input_image = Image.open(filename).convert('RGB')
            with torch.no_grad():
                output = model(preprocess_test(input_image).unsqueeze(0).cuda())

            _, predicted = torch.max(output.data, 1)
            if predicted[0]==NUM_CLASSES_DICT[cat]:
                predicted = [np.random.randint(0, NUM_CLASSES_DICT[cat])]
                df.at[i, 'is_face'] = True
            else:
                predicted = predicted.cpu()

            predicted_str = label_encoder.inverse_transform(predicted)
            df.at[i, cat] = predicted_str[0]

    print(df.head(10))
    df.to_csv(parent_fldr + 'labels_face_selflabel.csv', index=False)
