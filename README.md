## Required Packages

```
numpy
pandas
scikit-learn
PIL
torch
torchvision
```

## Quick Start and Inference

### Download trained model checkpoints

- Create a folder `models` in the home directory.

- Download all 3 checkpoints from [here](https://drive.google.com/drive/folders/1kX0WU5DdZl5ai54EB3XWO2BgDLaCdS0w?usp=share_link) and place them in the folder `models`.

### Execute Inference

Simply run inference with the following command,

_Note: Adjust batch size according to the GPU memory to maximize inference speed_

```
python inference.py --batch_size 64 --savefldr models/ --cuda --label_file data/test/labels.csv --img_parent_dir data/test/ --outfile outscore.json
```

Change the path to the label file and image parent directory if the inference is required to be done on a different folder.

### Insights on Model Prediction

The models used in this repo have classification heads that classify each image into pre-defined classes for a given demographic, and an additional background class. The background class here refers to identifying the absence of face in the image. The final predictions do not contain this 'background class', and instead the predictions for such images are replaced with random predictions. Thus, the inference process is non-deterministic, and one can expect minor variance in scores obtained across multiple runs.

## Training the Model Locally

### Setting Up The Dataset

#### Download Dataset and Extract Zip File

- Download the data zip file and extract it in the folder `data/`

Directory structure

```
parent_dir
├── data
│   ├── train
|   |   ├── labels.csv
|   |   ├── TRAIN00001.png
|   |   ├── TRAIN00002.png
|   |   ├── ...
│   ├── test
|   |   ├── labels.csv
|   |   ├── TEST0001.png
|   |   ├── TEST0002.png
|   |   ├── ...
│   └── README.txt
```

#### Add Face Labels to Identify Outliers

- We will use the `MTCNN` model from `facenet_pytorch` library to differentiate between face and not face images (Find library here : https://github.com/timesler/facenet-pytorch).

- Simply run the following commands to create new label files, i.e. `labels_face.csv`, for both train and test. We are also creating the labels for the public test set because we use it for our local validation, and then train on these labels as well for our final submission.

```
python is_face_check.py data/train/
python is_face_check.py data/test/
```

### Train Model On The Dataset

#### Download pretrained model on VGGFace2

Unfortunately, `VGGFace2` dataset is not available anymore, removed by the original authors. However, a few pre-trained models are indeed available and can be found [here](https://github.com/cydonia999/VGGFace2-pytorch)

- We download the pretrained model `resnet50_ft` for our setup. Direct link to the model [here](https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view)

- Place the pretrained model inside a folder named `pretrained/`.

#### Start training

Train the model with the following command

```
python train.py --batch_size 64 --epochs 30 --lr 1e-3 --cuda --savefldr models/
```

_Note : The models already present will get rewritten if the same savefldr is provided. Make sure to change the savefldr name when training a new model._
