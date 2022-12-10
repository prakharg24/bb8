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

## Details of the approach

A few important tricks used for the competition that gave noticeable improvements,

- Starting with a pre-trained model : Since we are working with a dataset that is relatively small, using pre-trained models was an important step to achieve a better model. I tried several datasets for pretraining, i.e., pretraining on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) classification, pretraining on [Digiface1M](https://microsoft.github.io/DigiFace1M/) with contrastive loss, and simply using pre-trained models on Imagenet-1k classification. However, for some reason that I didn't get a chance to fully explore, the pre-trained model on VGGFace2 available [here](https://github.com/cydonia999/VGGFace2-pytorch) gave the best performance. Unfortunately, the VGGFace2 dataset is not publicly available anymore, so I was limited to only the architectures already available.
- Data augmentation : Instead of manually choosing the correct set of augmentations, I simply used [AugMix](https://github.com/google-research/augmix) and it gave significant improvement over any manually chosen set of augmentations that I was using. There are other similar techniques of automatically choosing the augmentation present in literature, but I didn't get a chance to play with them.
- Identifying noisy labels : The competition had an explicit setup where a big portion of data points were not faces! Not only was this a challenge to handle during validation (to pass the randomness test on such inputs), but it also polluted the training set. I simply used the `MTCNN` model from [`facenet_pytorch`](https://github.com/timesler/facenet-pytorch) library to first differentiate between face and not face images for all data points publicly available. Once we had this distinction, I added a 'background' class to my model during training and prediction, such that the model itself can differentiate between say predicting the skin tone of the face, or simply saying that the image is actually not even a face. Identifying noisy labels in this setting was easy, since we knew exactly what kind of noise we were looking for. However, this line of research on simply identifying noisy examples has been getting some attention in the last few years (a very exciting work [here](https://metadata-archaeology.github.io/))
- Chi square test during training : I didn't simply submit the best model from training based on only accuracy and/or disparity, but instead also did chi square test and included that in the 'score' to make sure I picked a model checkpoint that worked best for the metric used by the compeititon.
- Using validation set for training : This was the last step for the final submission. For all previous steps, the validation set was kept separate to make sure that the choices made are not influenced by overfitting. However, once I was sure about the exact pipeline I wanted to use, I simply included the validation set into the training set to train a model on a bigger dataset.

_Note : The last point mentioned above was not verified properly. That is, I didn't get a chance to truly check whether simply training on the whole set with no validation to do early stopping was a good choice or not. My intuition says that the improvement with an increase in dataset should be better than the loss due to overfitting that might occur, but I didn't get a chance to verify this properly._
