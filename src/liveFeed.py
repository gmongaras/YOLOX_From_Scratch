from YOLOX import YOLOX
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image, ImageOps
import random
import torch
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import json
import cv2










def liveFeed():
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
    removal_threshold = 0.75# The threshold of predictions to remove if the
                            # confidence in that prediction is below this value
    score_thresh = 0.7      # The score threshold to remove boxes. If the score is
                            # less than this value, remove it
    IoU_thresh = 0.1        # The IoU threshold to update scores. If the IoU is
                            # greater than this value, update it's score
    
    
    # Model Loading Parameters
    loadDir = "../models"       # The directory to load the model from
    paramLoadName = "modelParams - test.json"   # File to load the model paramters from
    loadName = "model - test.pkl"  # Filename to load the model from
    
    
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
    
    
    
    
    ### Model Loading ###
    
    # Load in the model parameters
    with open(os.path.join(loadDir, paramLoadName), "r", encoding='utf-8') as f:
        data = json.load(f)
        
    # Save the loaded data as model paramters
    ImgDim = data['ImgDim']
    numCats = data['numCats']
    category_Ids = data['category_Ids']
    category_Ids_rev = {category_Ids[i]:i for i in category_Ids.keys()}
    
    
    
    # We don't care about the gradients
    with torch.no_grad():
        
        # Create the model
        model = YOLOX(device, numEpochs, 1, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_consts, reg_weight, category_Ids, removal_threshold, score_thresh, IoU_thresh)
        
        # Load the model from a saved state
        model.loadModel(loadDir, loadName, paramLoadName)
        
        
    
    
    ### Video Capture ###
    
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    # Iterate until ENTER or esc is pressed
    c = 0
    while c != 13 and c != 27:
        
        # Get a webcam state
        ret, frame = cap.read()
        
        # Resize the image
        img = Image.fromarray(frame) # Convert to PIL object
        img = ImageOps.mirror(img) # Mirrow image horizontally
        img = img.convert("RGB")   # Convert to RGB
        prop = ImgDim/(max(img.height, img.width)) # proportion to resize image
        new_h = round(img.height*prop)
        new_w = round(img.width*prop)
        img = img.resize((new_w, new_h))
        img = np.array(img)
        
        # Pad with zeros if needed
        img = np.pad(img, (((ImgDim-img.shape[0])-(ImgDim-img.shape[0])//2, ((ImgDim-img.shape[0])//2)), ((ImgDim-img.shape[1])-(ImgDim-img.shape[1])//2, ((ImgDim-img.shape[1])//2)), (0, 0)), mode='constant')
        
        
        # Convert the image to a tensor
        img_tensor = torch.tensor(np.array([img]), requires_grad=False, dtype=torch.float, device=device)
        

        
        # Get the predictions from the model
        with torch.no_grad():
            cls_p, reg_p, obj_p = model.predict(img_tensor, 1)
        
        # Iterate over all bounding boxes for this image
        for bbox in range(0, len(cls_p[0])):
            # Get the bounding box info
            cls_i = cls_p[0][bbox]
            reg_i = reg_p[0][bbox]
            obj_i = obj_p[0][bbox]
            
            # Create a Rectangle patch
            top_left = (round(reg_i[0].item()), round(reg_i[1].item()))
            bottom_right = (round((reg_i[0]+reg_i[2]).item()), round((reg_i[1]+reg_i[3]).item()))
            color = (0, 0, 255)
            img = cv2.rectangle(img, top_left, bottom_right, color, 2)
            
            # Create a text patch
            img = cv2.putText(img, f"{category_Ids_rev[cls_i]}    {obj_i}", (round(reg_i[0].item()), round(reg_i[1].item())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        # Resize the image
        img = cv2.resize(img, (600,600), interpolation=cv2.INTER_AREA)
        
        # Show the frame
        cv2.imshow('Input', img)
        c = cv2.waitKey(1)
    
    
    # Free the camera
    cap.release()
    cv2.destroyAllWindows()




if __name__=='__main__':
    liveFeed()