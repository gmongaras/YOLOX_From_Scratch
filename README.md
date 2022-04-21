# YOLOX_From_Scratch
Attempting to build YOLOX from scratch



# Data
I used the COCO dataset (2017 test and 2017 val):
https://cocodataset.org/#download

## Coco bounding box data format (what is each part of the 4-part bounding box?)
1: horizontal (x) value from left
2: verticle (y) value from top
3: width of bounding box
4: height of bounding box
- 1 and 2 define the top left region of the bounding box
- 3 and 4 define the width of the bounding box



# Things Read
original paper - https://arxiv.org/abs/2107.08430v2

yolov3 - https://arxiv.org/abs/1804.02767v1

OTA - https://arxiv.org/abs/2103.14259

focal loss - https://maxhalford.github.io/blog/lightgbm-focal-loss/#first-order-derivative