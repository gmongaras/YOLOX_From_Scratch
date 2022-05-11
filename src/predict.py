from YOLOX import YOLOX
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import random
import torch
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import json










def predict():
    # Model Hyperparameters
    device = "gpu"          # The device to train the model with (cpu or gpu)
    numEpochs = 300         # The number of epochs to train the model for
    warmupEpochs = 5        # Number of warmup epochs to train the model for
    lr_init = 0.01          # Initial learning rate
    weightDecay = 0.0005    # Amount to decay weights over time
    momentum = 0.9          # Momentum of the SGD optimizer
    ImgDim = 256            # The height and width dimension to convert FPN 
                            # (Feature Pyramid Network) encodings to
    numCats = 3             # The number of categories to predict from
    reg_consts = (          # The contraints on the regression size
        0, 64, 128, 256     # Basically constraints on how large the bounding
        )                   # boxes can be for each level in the network

    
    
    # Bounding Box Filtering Parameters
    removal_threshold = 0.5 # The threshold of predictions to remove if the
                            # confidence in that prediction is below this value
    score_thresh = 0.5      # The score threshold to remove boxes. If the score is
                            # less than this value, remove it
    IoU_thresh = 0.25       # The IoU threshold to update scores. If the IoU is
                            # greater than this value, update it's score
    
    
    # Training Paramters
    dataDir = "../testData" # Directory to load data from
    batchSize = 0           # The size of each minibatch of data (use 0
                            # to use a single batch)
    
    
    # Model Loading Parameters
    loadDir = "../models"       # The directory to load the model from
    paramLoadName = "modelParams - t.json"   # File to load the model paramters from
    loadName = "model - t.pkl"  # Filename to load the model from
    
    
    # Loss Function Hyperparameters
    FL_alpha = 4            # The focal loss alpha parameter
    FL_gamma = 2            # The focal loss gamma paramter
    reg_weight = 5.0        # Percent to weight regression loss over other loss
    
    
    
    # Putting device on GPU or CPU
    if device.lower() == "gpu":
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("GPU not available, defaulting to CPU\n")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')
    
    
    
    
    
    ### Data Loading ###
    
    # Ensure the directory exists
    assert os.path.isdir(loadDir), f"Load directory {loadDir} does not exist."
    
    # Ensure the parameter file exists
    assert os.path.isfile(os.path.join(loadDir, paramLoadName)), f"Load file {os.path.join(loadDir, paramLoadName)} does not exist."
    
    # Load in the model parameters
    with open(os.path.join(loadDir, paramLoadName), "r", encoding='utf-8') as f:
        data = json.load(f)
        
    # Save the loaded data as model paramters
    ImgDim = data['ImgDim']
    numCats = data['numCats']
    category_Ids = data['category_Ids']
    category_Ids_rev = {category_Ids[i]:i for i in category_Ids.keys()}
    
    
    # List to hold the images as tensors
    imgs = []
    
    # Iterate over all files in the directory
    for file in os.listdir(dataDir):
        # Load in the image
        img = io.imread(os.path.join(dataDir, file))
        
        # Resize the image
        img = Image.fromarray(img) # Convert to PIL object
        img = img.convert("RGB")   # Convert to RGB
        prop = ImgDim/(max(img.height, img.width)) # proportion to resize image
        new_h = round(img.height*prop)
        new_w = round(img.width*prop)
        img = img.resize((new_w, new_h))
        img = np.array(img)
        
        # Pad with zeros if needed
        img = np.pad(img, (((ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, ((ImgDim-img.shape[0])//2)), ((ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2, ((ImgDim-img.shape[1])//2)), (0, 0)), mode='constant')
        
        # Save the images as a tensor
        imgs.append(torch.tensor(img, dtype=torch.float, requires_grad=False, device=device))
    
    # Save all the images as a single tensor
    imgs = torch.stack(imgs).to(device)
    
    
    
    
    ### Model Loading ###
    
    # We don't care about the gradients
    with torch.no_grad():
        
        # Create the model
        model = YOLOX(device, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_consts, reg_weight, category_Ids, removal_threshold, score_thresh, IoU_thresh)
        
        # Load the model from a saved state
        model.loadModel(loadDir, loadName, paramLoadName)
        
        # Get the predictions from the model
        cls_p, reg_p, obj_p = model.predict(imgs, batchSize)
    
    
    
    # Convert the images to a numpy array
    imgs = imgs.cpu().numpy()
    
    # Iterate over all images
    for img_num in range(0, imgs.shape[0]):
        # Get the image as a Pillow object
        img = Image.fromarray(np.uint8(imgs[img_num]))
        
        # Create the figure and subplots
        fig, ax = plt.subplots()
        
        # Display the image
        ax.imshow(img)
        
        # Iterate over all bounding boxes for this image
        for bbox in range(0, len(cls_p[img_num])):
            # Get the bounding box info
            cls_i = cls_p[img_num][bbox]
            reg_i = reg_p[img_num][bbox]
            obj_i = obj_p[img_num][bbox]
            
            # Create a Rectangle patch
            rect = patches.Rectangle((reg_i[0], reg_i[1]), reg_i[2], reg_i[3], linewidth=1, edgecolor='r', facecolor='none')
        
            # Add the rectangle to the image
            ax.add_patch(rect)
            
            # Create a text patch
            plt.text(reg_i[0], reg_i[1], f"{category_Ids_rev[cls_i]}    {obj_i}", fontdict=dict(fontsize="xx-small"), bbox=dict(fill=False, edgecolor='red', linewidth=0))
        
        plt.show()




if __name__=='__main__':
    predict()