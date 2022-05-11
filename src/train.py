from YOLOX import YOLOX
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import random
import torch
import math
from matplotlib import pyplot as plt
import os










def train():
    # Model Hyperparameters
    device = "gpu"          # The device to train the model with (cpu or gpu)
    numEpochs = 300         # The number of epochs to train the model for
    batchSize = 128         # The size of each minibatch
    warmupEpochs = 5        # Number of warmup epochs to train the model for
    lr_init = 0.01          # Initial learning rate
    weightDecay = 0.0005    # Amount to decay weights over time
    momentum = 0.9          # Momentum of the SGD optimizer
    reg_consts = (          # The contraints on the regression size
        0, 64, 128, 256     # Basically constraints on how large the bounding
        )                   # boxes can be for each level in the network
    ImgDim = 256            # Resize the images to a quare pixel value (can be 1024, 512, or 256)
    
    
    
    # Bounding Box Filtering Parameters
    removal_threshold = 0.5 # The threshold of predictions to remove if the
                            # confidence in that prediction is below this value
    score_thresh = 0.5      # The score threshold to remove boxes. If the score is
                            # less than this value, remove it
    IoU_thresh = 0.25       # The IoU threshold to update scores. If the IoU is
                            # greater than this value, update it's score
    
    
    # SimOta Parameters
    q = 20              # The number of GIoU values to pick when calculating the k values
                        #  - k = The number of labels (supply) each gt has
    r = 5               # The radius used to calculate the center prior
    extraCost = 100000  # The extra cost used in the center prior computation
    SimOta_lambda = 3   # Balancing factor for the foreground loss
    # Note: The best values for q, r, and lambda are chosen above
    
    
    # Model Save Parameters
    saveDir = "../models"   # The directory to save models to
    saveName = "model"      # File to save the model to
    paramSaveName = "modelParams"   # File to save the model parameters to
    saveSteps = 10          # Save the model every X steps
    saveOnBest = False      # Save the model only if it's the
                            # best model at save time
    overwrite = False       # True to overwrite the existing file when saving.
                            # False to make a new file when saving.
    
    
    # Model Loading Parameters
    loadModel = False           # True to load in a pretrained model, False otherwise
    loadDir = "../models"       # The directory to load the model from
    paramLoadName = "modelParams.json"   # File to load the model paramters from
    loadName = "model.pkl"  # Filename to load the model from
    
    # Loss Function Hyperparameters
    FL_alpha = 4            # The focal loss alpha parameter
    FL_gamma = 2            # The focal loss gamma paramter
    reg_weight = 5.0        # Percent to weight regression loss over other loss
    
    
    # COCO dataset parameters
    dataDir = "../coco"     # The location of the COCO dataset
    dataType = "val2017"    # The type of data being used in the COCO dataset
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    categories = []         # The categories to load in (empty list to load all)
    numToLoad = 10          # Max Number of data images to load in (use -1 for all)
    
    
    
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
    
    
    
    ### Data Loading
    
    # Initialize the COCO data
    coco=COCO(annFile)
    
    
    # Get the number of categories
    numCats = len(categories)
    
    
    # If the number of categories is 0, load all categories
    if numCats == 0:
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
        #img = io.imread(img_d['coco_url'])
        
        # Resize the image
        img = Image.fromarray(img) # Convert to PIL object
        img = img.convert("RGB")   # Convert to RGB
        prop = ImgDim/(max(img.height, img.width)) # proportion to resize image
        new_h = round(img.height*prop)
        new_w = round(img.width*prop)
        img = img.resize((new_w, new_h))
        #img.thumbnail((new_h, new_w), Image.ANTIALIAS) # Resize the image
        img = np.array(img)
        
        # Pad with zeros if needed
        pad = [(ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, (ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2]
        img = np.pad(img, (((ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, ((ImgDim-img.shape[0])//2)), ((ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2, ((ImgDim-img.shape[1])//2)), (0, 0)), mode='constant')
        
        # Save the array, the resize proportion, and the padding
        imgs.append(img)
        props.append(prop)
        padding.append(pad)
    imgs = torch.tensor(np.array(imgs), dtype=torch.float32, requires_grad=False, device=cpu)
    
    # Correction so that channels are first, not last
    if imgs.shape[-1] == 3:
        imgs = torch.reshape(imgs, (imgs.shape[0], 3, imgs.shape[1], imgs.shape[2]))
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
        ann = ann_data[img_d["id"]]
        ann_bbox = []
        ann_cls = []
        
        # The class for each pixel in the image
        pix_cls = np.zeros((img.shape[1], img.shape[2]))
        
        # The object probability for each pixel in the image
        # (1 if object in pixel, 0 otherwise)
        pix_obj = np.zeros((img.shape[1], img.shape[2]))
        
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
                        pix_obj[:][w][h] = 1
        
        # Encode the classes as a tensor
        pix_cls = torch.tensor(pix_cls, dtype=int, device=cpu, requires_grad=False)
        
        # One hot encode the pixel classes
        #pix_cls = torch.nn.functional.one_hot(torch.tensor(pix_cls, dtype=int, device=cpu), len(seq_category_Ids.values()))
        
        # Encode the objectiveness as a tensor
        pix_obj = torch.tensor(pix_obj, dtype=int, device=cpu, requires_grad=False)
        
        anns.append({"bbox":ann_bbox, "cls":ann_cls, "pix_cls":pix_cls, "pix_obj":pix_obj})
    print("Annotations Loaded!\n")
    
    
    
    
    ### Model Training ###
    
    # File saving parameters
    saveParams = [saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite]
    
    # SimOta Paramters
    SimOTA_params = [q, r, extraCost, SimOta_lambda]
    
    # Create the model
    #torch.autograd.set_detect_anomaly(True)
    model = YOLOX(device, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_consts, reg_weight, seq_category_Ids, removal_threshold, score_thresh, IoU_thresh, SimOTA_params)
    
    # Load the model if requested
    if loadModel:
        model.loadModel(loadDir, loadName, paramLoadName)
    
    # Train the model
    model.train_model(imgs, anns, saveParams)
    
    
    #### Model Testing
    print()
    
    # To get output, use torchvision.ops.nms
    
    # Make inferences using
    # with torch.no_grad():
    #    make_inference()
    
    
    
    ### Model Saving
    
    # Save model in pkl file
    
    # Save model parameters (like the layers sizes and such) to load
    # in for inferring




if __name__=='__main__':
    train()