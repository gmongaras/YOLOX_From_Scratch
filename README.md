# TODO
- Add L1 Loss (found in YoloX source code)
- Model crashes after about 150 iters on 128 data points. Is this fixed with obj loss fix?
- SimOTA for label assignments (probably only used during training)
- Data Augmentation (Mosaic and MixUp). Don't use these augmentations for the last 15 iterations.


# Contents
- [Project Purpose](#project-purpose)
- [Model Information](#model-information)
- Coco Dataset Information
- Running The Model
  - Trianing
  - Predicting
  - Live Feed
- Results
- Sources



# Project Purpose

The purpose of this project is to code the YOLOX algorithm from scratch in order to learn more about how it works and to put the algorithm in a more readable format for other people to better understand the algorithm. 

When reading over the YOLOX paper, I noticed it was missing a lot of content which was assumed knowledge from other papers like YOLOv3, OTA, FCOS, and others. Since this algorithm does better than the famous YOLO algorithms but does so without anchors, it is important to understand how it works in order to improve bounding box algorithms in an anchor-free manor. Using this repo, I will attempt to explain how the algorithm works in some sort of article format and will put the links below as I write them:

Links to come...


# Model Information

What problem is the model trying to solve?
This model is the first YOLO algorithm to use anchor-free detection. An anchor is basically a predefined bounding box shape that helps the network. Instead of predicting the direct bounding box, previous YOLO algorithms predicted an offset from a predefined anchor box. So, if an anchor box had length and with of 100 and 50 and the model predicted length and width of 10 and 15, the bounding box prediction would be an offset from the anchor box with a length and width of 110 and 65. More information about anchor boxes can be found [in this conversation](https://github.com/pjreddie/darknet/issues/568).

What's the problem with anchor boxes?
Anchor boxes are basically an extra parameter. How anchors should the model use? What should the sizes of the anchors be? These questions lead to more hyperparameter tuning and less diversity in the model. 

How does the model solve the anchor box problem?
YOLOX simply has the model directly predict the bounding box dimensions as opposed to predicting an offset from an anchor box. To do this, it uses a decoupled head unlike other YOLO algorithms. Below is the side by side comparison between the YOLOv3 model and the YOLOX model which can be found in the YOLOX paper:

<p align="center">
  <img src=https://user-images.githubusercontent.com/43501738/167199518-17bdf353-1636-493e-b937-9e3c6f8a2349.png>
</p>

The final predictions of this model are the following:
- reg - The predicted bounding box which has 4 values:
  1. X value of the top-left corner of the bounding box
  2. Y value of the top-left corner of the bounding box
  3. Height of the boudning box
  4. Width of the bounding box
- cls - The predicted class the model thinks is inside the bounding box. This is a one-hot encoded vector with the same number of elements are there are classes.
- IoU (obj) - The objectiveness prediction of the predicted bounding box. This is a single value showing how confident an object is in the predicted bounding box.

More about the model can be found in the [articles I wrote](#project-purpose).


# Coco Dataset Information
I used the COCO dataset (2017 test and 2017 val):
https://cocodataset.org/#download

## Coco bounding box data format (what is each part of the 4-part bounding box?)
1. horizontal (x) value from left
2. verticle (y) value from top
3. width of bounding box
4. height of bounding box
- 1 and 2 define the top left region of the bounding box
- 3 and 4 define the width of the bounding box



# Things Read
original paper - https://arxiv.org/abs/2107.08430v2

yolov3 - https://arxiv.org/abs/1804.02767v1

OTA - https://arxiv.org/abs/2103.14259

focal loss - https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872

nonmax supression - https://arxiv.org/pdf/1704.04503.pdf

GIoU - https://giou.stanford.edu/
