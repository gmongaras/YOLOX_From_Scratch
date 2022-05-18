from YOLOX import YOLOX
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import torch
import math
import os
from copy import deepcopy
import click
from typing import List, Optional
import re





def num_range(s):
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.
    """

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def str_to_list(s):
    """
    Convert a string of form 'a,b,c' to a list ['a', 'b', 'c']
    """
    return s.replace(" ", "").split(",")
    






@click.command()
# Required
@click.option("--dataDir", "dataDir", type=str, help="Location of the COCO dataset", required=True)
@click.option("--dataType", "dataType", type=str, help="The type of data being used in the COCO dataset (ex: val2017)", required=True)
@click.option("--numToLoad", "numToLoad", type=int, help="Max Number of data images to load in (use -1 for all)", required=True)

# Hyperparamters
@click.option("--device", "device", type=str, default="gpu", help="The device to train the model with (cpu or gpu)", required=False)
@click.option("--numEpochs", "numEpochs", type=int, default=300, help="The number of epochs to train the model", required=False)
@click.option("--batchSize", "batchSize", type=int, default=128, help="The size of each minibatch", required=False)
@click.option("--warmupEpochs", "warmupEpochs", type=int, default=5, help="Number of epochs before using a lr scheduler", required=False)
@click.option("--alpha", "alpha", type=float, default=0.01, help="Initial learning rate", required=False)
@click.option("--weightDecay", "weightDecay", type=float, default=0.0005, help="Weight decay in SGD", required=False)
@click.option("--momentum", "momentum", type=float, default=0.9, help="Momentum of SGD", required=False)
@click.option("--ImgDim", "ImgDim", type=int, default=256, help="Resize the images to a square pixel value (can be 1024, 512, or 256)", required=False)
@click.option("--augment_per", "augment_per", type=float, default=0.75, help="Percent of extra augmented data to generate every epoch", required=False)

# Bounding Box Filtering
@click.option("--removal_threshold", "removal_threshold", type=float, default=0.5, help="The threshold of predictions to remove if the confidence in that prediction is below this value", required=False)
@click.option("--score_thresh", "score_thresh", type=float, default=0.5, help="The score threshold to remove boxes in NMS. If the score is less than this value, remove it", required=False)
@click.option("--IoU_thresh", "IoU_thresh", type=float, default=0.1, help="The IoU threshold to update scores in NMS. If the IoU is greater than this value, update it's score", required=False)

# SimOTA Paramters
@click.option("--q", "q", type=int, default=20, help="The number of GIoU values to pick when calculating the k values in SimOTA (k = The number of labels (supply) each gt has)", required=False)
@click.option("--r", "r", type=int, default=5, help="The radius used to calculate the center prior in SimOTA", required=False)
@click.option("--extraCost", "extraCost", type=float, default=100000.0, help="The extra cost used in the center prior computation in SimOTA", required=False)
@click.option("--SimOta_lambda", "SimOta_lambda", type=float, default=3.0, help="Balancing factor for the foreground loss in SimOTA", required=False)

# Model Save Paramters
@click.option("--saveDir", "saveDir", type=str, default="../models", help="The directory to save models to", required=False)
@click.option("--saveName", "saveName", type=str, default="model", help="File to save the model to", required=False)
@click.option("--paramSaveName", "paramSaveName", type=str, default="modelParams", help="File to save the model parameters to", required=False)
@click.option("--saveSteps", "saveSteps", type=int, default=10, help="Save the model every `saveSteps` steps", required=False)
@click.option("--saveOnBest", "saveOnBest", type=bool, default=False, help="True to save the model only if it's the current best model at save time", required=False)
@click.option("--overwrite", "overwrite", type=bool, default=False, help="True to overwrite the existing file when saving. False to make a new file when saving.", required=False)

# Model Loading Paramters
@click.option("--loadModel", "loadModel", type=bool, default=False, help="True to load in a pretrained model, False otherwise", required=False)
@click.option("--loadDir", "loadDir", type=str, default="../models", help="The directory to load the model from", required=False)
@click.option("--paramLoadName", "paramLoadName", type=str, default="modelParams.json", help="File to load the model paramters from", required=False)
@click.option("--loadName", "loadName", type=str, default="model.pkl", help="Filename to load the model from", required=False)

# Loss Function Hyperparamters
@click.option("--FL_alpha", "FL_alpha", type=float, default=4.0, help="The focal loss alpha parameter", required=False)
@click.option("--FL_gamma", "FL_gamma", type=float, default=2.0, help="The focal loss gamma paramter", required=False)
@click.option("--reg_weight", "reg_weight", type=float, default=5.0, help="Percent to weight regression loss over other loss", required=False)

# Coco dataset paramters
@click.option("--categories", "categories", type=str_to_list, default="", help="The categories to load in (empty list to load all) (Ex: 'cat,dog,person'", required=False)

def train(
    dataDir: str,
    dataType: str,
    numToLoad: int,

    device: Optional[str],
    numEpochs: Optional[int],
    batchSize: Optional[int],
    warmupEpochs: Optional[int],
    alpha: Optional[float],
    weightDecay: Optional[float],
    momentum: Optional[float],
    ImgDim: Optional[int],
    augment_per: Optional[float],

    removal_threshold: Optional[float],
    score_thresh: Optional[float],
    IoU_thresh: Optional[float],

    q: Optional[int],
    r: Optional[int],
    extraCost: Optional[float],
    SimOta_lambda: Optional[float],

    saveDir: Optional[str],
    saveName: Optional[str],
    paramSaveName: Optional[str],
    saveSteps: Optional[int],
    saveOnBest: Optional[bool],
    overwrite: Optional[bool],

    loadModel: Optional[bool],
    loadDir: Optional[str],
    paramLoadName: Optional[str],
    loadName: Optional[str],

    FL_alpha: Optional[float],
    FL_gamma: Optional[float],
    reg_weight: Optional[float],

    categories: Optional[List[str]],
    
    ):
    
    # The annotations file for the COCO dataset
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
    
    
    # Putting device on GPU or CPU
    if device.lower() == "gpu":
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("GPU not available, defaulting to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')
    
    
    
    
    
    
    ### Data Loading ###
    
    # Initialize the COCO data
    coco=COCO(annFile)
    
    
    # Get the number of categories
    numCats = len(categories)
    
    
    # If the number of categories is 0, load all categories
    if numCats == 0 or categories[0] == '':
        categories = [coco.cats[i]['name'] for i in coco.cats]
        numCats = len(categories)
    
    
    
    # Get the image data and annotations
    img_data = []
    category_Ids = dict()
    ann_data = dict()
    seen = []
    for c in categories:
        # Get all images in the category c
        catIds = coco.getCatIds(catNms=[c])
        imgIds = coco.getImgIds(catIds=catIds)
        sub_img_data = coco.loadImgs(imgIds)
        
        # Save the category ids
        category_Ids[c] = catIds[0]
        
        # Get the new images
        notSeen = []
        for i in sub_img_data:
            if i['id'] not in seen:
                seen.append(i['id'])
                notSeen.append(i)
        img_data += notSeen
        
        # Save the annotations for the new images
        for i in notSeen:
            ann_data[i['id']] = coco.loadAnns(coco.getAnnIds(imgIds=i['id'], iscrowd=None))
    
    # Get a subset if requested
    if numToLoad > 0:
        img_data = img_data[:numToLoad]
    
    
    # Load in the actual images
    imgs = []
    props = [] # Proportions to resize the images
    padding = [] # Amount of padding added to the image
    print("\nLoading images...")
    for img_d in img_data:
        # Load in the image
        img = io.imread(dataDir + os.sep + "images" + os.sep + dataType + os.sep + img_d["file_name"])
        
        # Resize the image
        img = Image.fromarray(img) # Convert to PIL object
        img = img.convert("RGB")   # Convert to RGB
        prop = ImgDim/(max(img.height, img.width)) # proportion to resize image
        new_h = round(img.height*prop)
        new_w = round(img.width*prop)
        img = img.resize((new_w, new_h))
        img = np.array(img)
        
        # Pad with zeros if needed
        pad = [(ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, (ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2]
        img = np.pad(img, (((ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, ((ImgDim-img.shape[0])//2)), ((ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2, ((ImgDim-img.shape[1])//2)), (0, 0)), mode='constant')
        
        # Save the array, the resize proportion, and the padding
        imgs.append(img)
        props.append(prop)
        padding.append(pad)
    imgs = torch.tensor(np.array(imgs), dtype=torch.int16,  requires_grad=False, device=cpu)
    
    # Correction so that channels are first, not last
    if imgs.shape[-1] == 3:
        imgs = imgs.permute(0, 3, 1, 2)
    print("Images Loaded")
    
    
    # Save the category ids as sequential numbers instead of
    # the default number given to that category
    seq_category_Ids = {list(category_Ids.keys())[i]:i+1 for i in range(0, len(category_Ids.keys()))}
    seq_category_Ids_rev = {i+1:list(category_Ids.keys())[i] for i in range(0, len(category_Ids.keys()))}
    seq_category_Ids["Background"] = 0
    seq_category_Ids_rev[0] = "Background"
    
    
    # Get all annoations which we want
    anns = []
    print("\nLoading Annotations...")
    for i in range(0, len(img_data)):
        img_d = img_data[i]
        img = imgs[i]
        prop = props[i]
        pad = padding[i]
        
        # Get all annotations for this image
        ann = deepcopy(ann_data[img_d["id"]])
        ann_bbox = []
        ann_cls = []
        
        # The class for each pixel in the image
        pix_cls = np.zeros((img.shape[1], img.shape[2]), dtype=np.int16)
        
        # Iterate over every annotation and save it into a better form
        for a in ann:
            # Save the annotation if it bounds the wanted object
            if a["category_id"] in category_Ids.values():
                # Get the bounding box
                bbox = a["bbox"]
                
                # Resize the bounding box
                for j in range(0, len(bbox)):
                    bbox[j] = bbox[j]*prop
                    
                # Add the padding to the bounding boxes
                bbox[0] += pad[1]
                bbox[1] += pad[0]
                
                # Save the bounding box
                ann_bbox.append(bbox)
                
                # Save the class
                cls = list(category_Ids.values()).index(a["category_id"])+1
                ann_cls.append(cls)
                
                # Round the bounding box values to get an integer
                # pixel value. The box is rounded to capture the
                # smallest amount of area.
                bbox[0] = math.floor(bbox[0])
                bbox[1] = math.floor(bbox[1])
                bbox[2] = math.ceil(bbox[2])
                bbox[3] = math.ceil(bbox[3])
                
                # Add the class and the object probability
                # to each pixel the bounding box captures
                # Note: probability is 1 if an object is in the
                #       part of the image and 0 otherwise
                for w in range(bbox[0], bbox[0]+bbox[2]):
                    for h in range(bbox[1], bbox[1]+bbox[3]):
                        pix_cls[:][w][h] = cls
        
        # Encode the classes as a tensor
        pix_cls = torch.tensor(pix_cls, device=cpu, requires_grad=False, dtype=torch.int16)
        
        anns.append({"bbox":ann_bbox, "cls":ann_cls, "pix_cls":pix_cls})
    print("Annotations Loaded!\n")
    
    
    
    
    
    ### Model Training ###
    
    # File saving parameters
    saveParams = [saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite]
    
    # SimOta Paramters
    SimOTA_params = [q, r, extraCost, SimOta_lambda]
    
    # Data augmentation paramters
    dataAug_params = [dataDir + os.sep + "images" + os.sep + dataType + os.sep, img_data, ann_data, category_Ids]
    
    # Create the model
    model = YOLOX(device, numEpochs, batchSize, warmupEpochs, alpha, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_weight, seq_category_Ids, removal_threshold, score_thresh, IoU_thresh, SimOTA_params)
    
    # Load the model if requested
    if loadModel:
        model.loadModel(loadDir, loadName, paramLoadName)
    
    # Train the model
    model.train_model(imgs, anns, dataAug_params, augment_per, saveParams)




if __name__=='__main__':
    try:
        train()
    except SystemExit:
        raise RuntimeError("Usage: python train.py --dataDir=[dataDir] --dataType=[dataType] --numToLoad=[numToLoad]")