# Contents
- [Project Purpose](#project-purpose)
- [Project Requirements](#project-requirements)
- [Model Information](#model-information)
- [Cloning This Repo](#cloning-this-repo)
  - [Directory Information](#directory-information)
- [Coco Dataset Information](#coco-dataset-information)
  - [Downloading The Data](#downloading-the-data)
  - [Coco Bounding Box Data Format](#coco-bounding-box-data-format)
- [Running The Model](#running-the-model)
  - [Training](#training)
  - [Predicting](#predicting)
  - [Live Feed](#live-feed)
- Results
- Sources


# Project Requirements

This project was written in Python. At the time of this README update, `python 3.8.10` was used, but any Python up to 3.9 should work.

The following libraries with their versions are needed to completely run this project:
```
PyTorch: 1.11.0
PyCocoTools: 2.0.4
NumPy: 1.22.3
SciKit Image: 0.18.2
Pillow: 8.2.0
Matplotlib: 3.4.2
CV2: 4.5.2.54
```

You can install all these libraries using the following commands:
```
pip install torch
pip install pycocotools
pip install numpy
pip install scikit-image
pip install pillow
pip install matplotlib
pip install opencv-python
```

To install PyTorch with Cuda support, go to the link below:
https://pytorch.org/



# Project Purpose

The purpose of this project is to code the YOLOX algorithm from scratch in order to learn more about how it works and to put the algorithm in a more readable format for other people to better understand the algorithm. 

The original paper can be found from the link below:
https://arxiv.org/abs/2107.08430

The original repo can be found using the link below:
https://github.com/Megvii-BaseDetection/YOLOX

When reading over the YOLOX paper, I noticed it was missing a lot of content that was assumed knowledge from other papers like YOLOv3, OTA, FCOS, and others. Since this algorithm does better than the famous YOLO algorithms but does so without anchors, it is important to understand how it works in order to improve bounding box algorithms in an anchor-free manner. Using this repo, I will attempt to explain how the algorithm works in some sort of article format and will put the links below as I write them:

[What is YOLO and What Makes It Special?](https://gmongaras.medium.com/yolox-explanation-what-is-yolox-and-what-makes-it-special-c01f6a8a0830)<br>
How Does YOLOX Work?<br>
SimOTA For Dynamic Label Assignment<br>
Mosaic and Mixup For Data Augmentation<br>


# Model Information

What problem is the model trying to solve?
This model is the first YOLO (You Only Look Once) algorithm to use anchor-free detection. An anchor is basically a predefined bounding box shape that helps the network. Instead of predicting the direct bounding box, previous YOLO algorithms predicted an offset from a predefined anchor box. So, if an anchor box had length and with of 100 and 50 and the model predicted length and width of 10 and 15, the bounding box prediction would be an offset from the anchor box with a length and width of 110 and 65. More information about anchor boxes can be found [in this conversation](https://github.com/pjreddie/darknet/issues/568).

What's the problem with anchor boxes?
Anchor boxes are basically extra parameters. How many anchors should the model use? What should the sizes of the anchors be? These questions lead to more hyperparameter tuning and less diversity in the model. 

How does the model solve the anchor box problem?
YOLOX simply has the model directly predict the bounding box dimensions as opposed to predicting an offset from an anchor box. To do this, it uses a decoupled head, unlike other YOLO algorithms. Below is the side by side comparison between the YOLOv3 model and the YOLOX model, which can be found in the YOLOX paper:

<p align="center">
  <img src=https://user-images.githubusercontent.com/43501738/167199518-17bdf353-1636-493e-b937-9e3c6f8a2349.png>
</p>

The final predictions of this model are the following:
- Reg - The predicted bounding box which has 4 values:
  1. X value of the top-left corner of the bounding box
  2. Y value of the top-left corner of the bounding box
  3. Height of the bounding box
  4. Width of the bounding box
- Cls - The predicted class the model thinks is inside the bounding box. This is a one-hot encoded vector with the same number of elements are there are classes.
- IoU (obj) - The objectiveness prediction of the predicted bounding box. This is a single value showing how confident an object is in the predicted bounding box.

More about the model can be found in the [articles I wrote](#project-purpose).



# Cloning This Repo

To clone this repo, use the following command in your computer terminal:
```
git clone https://github.com/gmongaras/YOLOX_From_Scratch.git
```

After cloning the repo, please download the Coco data which is specified in [Coco Dataset Information](#coco-dataset-information)

## Directory Information

The Coco data should be in the proper format so the model can properly find the data. The directory tree below is how the repo should be properly formatted, assuming the val 2017 and test 2017 data were downloaded:

```
.
├── coco
│   ├── annotations
|   |   ├── captions_train2017.json
|   |   ├── captions_val2017.json
|   |   ├── instances_train2017.json
|   |   ├── instances_val2017.json
|   |   ├── person_keypoints_train2017.json
|   |   └── person_keypoints_val2017.json
│   └── images
|   ├── train 2017
|   |   ├── 000000000009.jpg
|   |   ├── 000000000025.jpg
|   |   ├── ...
|   |   └── {more images in the train2017 dataset}
|   ├── val2017
|   |   ├── 000000000139.jpg
|   |   ├── 000000000285.jpg
|   |   ├── ...
|   |   └── {more images in the val2017 dataset}
├── models
│   ├── model - test.pkl
|   └── modelParams - test.json
├── src
│   ├── YOLOX.py
│   └── {all other .py scripts}
├── testData
|   ├── 000000013201.jpg
|   └── {Other images to test on}
├── .gitignore
└── README.md
```



# Coco Dataset Information

To test and train this particular model, I used the Coco dataset. The Coco dataset has images along with bounding box labels in those images. More information can be found on the [Coco website](https://cocodataset.org/#home).

## Downloading The Data

In particular, I used the 2017 val and 2017 train data to train/test this model. The data can be found at the following link: https://cocodataset.org/#download

Direct download links can be found below:
- Note: The data takes up about 20 Gb of memory.

1. Uncompress the following and place all images in the `./coco/images/train2017/` directory:
http://images.cocodataset.org/zips/test2017.zip

2. Uncompress the following and place all images in the `./coco/images/val2017/` directory:
http://images.cocodataset.org/zips/val2017.zip

3. Uncompress the following and place all annotations in the `./coco/annotations/` directory:
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After downloading the data, your filesystem should look [like the following](#directory-information).


## Coco Bounding Box Data Format

Each bounding box has 4 values. These values line up with what we want our model to predict which are:
1. horizontal (x) value from left
2. verticle (y) value from top
3. width of the bounding box
4. height of the bounding box
- 1 and 2 define the top left region of the bounding box
- 3 and 4 define the length and width of the bounding box


# Pretrained Models

Pretrianed models can be found using the following google drive link:
https://drive.google.com/drive/folders/1hXQQgntAAs0DdrcaF8FtR4_nZnhMyvZb?usp=sharing

Please ensure that any models that were downloaded are paired with their parameters. Each model has two files:
1. A .pkl file which stores the model data
2. A .json file that stores extra configuration information on the model

Both files should go into the `./models/` Directory within your local repository.

After the model has been downloaded, ensure the filesystem [looks like the following](#directory-information).


# Running The Model

There are three different scripts I wrote to run the model:

## Training

To train the model, ensure the [data was downloaded correctly](#downloading-the-data).

To train the model using a pre-trained model, ensure a [pretrained model was downloaded](#pretrained-models).

Assuming you now have the data and an optional pre-trained model on your computer, use the following command from the root directory of this repository to begin training the model:

```
cd src/
python train.py
```

More info coming in the future

## Predicting

To make predictions with the model, ensure a [pretrained model was downloaded](#pretrained-models).

Additionally, any images you wish the model to put bounding boxes around should be placed into the `./testData/` directory of this repository. A couple of images are already supplied.

Assuming the pre-trained model was downloaded and is in the correct repository, use the following command from the root directory of this repository to begin making predictions with the model:

```
cd src/
python predict.py
```


## Live Feed

The Live Feed mode will use a pre-trained model and your device camera to put bounding boxes around your camera environment in real-time.

To use the live feed mode, ensure a [pretrained model was downloaded](#pretrained-models).

To run the live feed mode, use the following command from the root repository in the directory:

```
cd src/
python liveFeed.py
```


To stop the live feed script, press `esc` or `ENTER`.


# Things Read
original paper - https://arxiv.org/abs/2107.08430v2

yolov3 - https://arxiv.org/abs/1804.02767v1

OTA - https://arxiv.org/abs/2103.14259

focal loss - https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872

nonmax suppression - https://arxiv.org/pdf/1704.04503.pdf

GIoU - https://giou.stanford.edu/

FCOS - https://arxiv.org/abs/1904.01355

MixUp - https://arxiv.org/pdf/1710.09412.pdf
