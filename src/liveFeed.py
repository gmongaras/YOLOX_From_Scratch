from YOLOX import YOLOX
import numpy as np
from PIL import Image, ImageOps
import torch
import os
import json
import cv2
import click
from typing import Optional










@click.command()
# Required
@click.option("--loadDir", "loadDir", type=str, help="The directory to load the model from", required=True)
@click.option("--paramLoadName", "paramLoadName", type=str, help="File to load the model parameters from", required=True)
@click.option("--loadName", "loadName", type=str, help="Filename to load the model from", required=True)

# Other Parameters
@click.option("--device", "device", type=str, default="gpu", help="The device to train the model with (cpu or gpu)", required=False)

# Bounding Box Filtering
@click.option("--removal_threshold", "removal_threshold", type=float, default=0.5, help="The threshold of predictions to remove if the confidence in that prediction is below this value", required=False)
@click.option("--score_thresh", "score_thresh", type=float, default=0.5, help="The score threshold to remove boxes in NMS. If the score is less than this value, remove it", required=False)
@click.option("--IoU_thresh", "IoU_thresh", type=float, default=0.1, help="The IoU threshold to update scores in NMS. If the IoU is greater than this value, update it's score", required=False)

# Focal Loss Function Hyperparameters
@click.option("--FL_alpha", "FL_alpha", type=float, default=4.0, help="The focal loss alpha parameter", required=False)
@click.option("--FL_gamma", "FL_gamma", type=float, default=2.0, help="The focal loss gamma parameter", required=False)
def liveFeed(
    loadDir: str,
    paramLoadName: str,
    loadName: str,
    
    device: Optional[str],

    removal_threshold: Optional[float],
    score_thresh: Optional[float],
    IoU_thresh: Optional[float],

    FL_alpha: Optional[float],
    FL_gamma: Optional[float],
    
    ):
    # Trash parameters the model requires but does not use
    numEpochs = 300
    warmupEpochs = 5
    lr_init = 0.01
    weightDecay = 0.0005
    momentum = 0.9
    FL_alpha = 4
    FL_gamma = 2
    reg_weight = 5
    
    # Trash parameters the model needs but will overwrite
    # using the model parameters file
    ImgDim = 256
    numCats = 3
    category_Ids = dict()
    
    
    
    # Putting device on GPU or CPU
    if device.lower() == "gpu":
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')
    
    
    
    
    ### Model Loading ###
    
    # Load in the model parameters
    with open(os.path.join(loadDir, paramLoadName), "r", encoding='utf-8') as f:
        data = json.load(f)
        
    # Store the loaded data as model parameters
    ImgDim = data['ImgDim']
    numCats = data['numCats']
    category_Ids = data['category_Ids']
    category_Ids_rev = {category_Ids[i]:i for i in category_Ids.keys()}
    
    
    
    # We don't care about the gradients
    with torch.no_grad():
        
        # Create the model
        model = YOLOX(device, numEpochs, 1, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_weight, category_Ids, removal_threshold, score_thresh, IoU_thresh)
        
        # Load the model from a saved state
        model.loadModel(loadDir, loadName, paramLoadName)
        
        
    
    
    ### Video Capture ###
    
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    # Iterate until ENTER or esc is pressed
    c = 0
    print("Press ENTER or esc to end the script")
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
    
    return 0




if __name__=='__main__':
    # Usage: python liveFeed.py --loadDir=[loadDir] --paramLoadName=[paramLoadName] --loadName=[loadName]
    liveFeed()