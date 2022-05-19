from YOLOX import YOLOX
import numpy as np
import skimage.io as io
from PIL import Image
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import json
import click
from typing import Optional











@click.command()
# Required
@click.option("--dataDir", "dataDir", type=str, help="Directory to load data we want the model to make predictions on", required=True)
@click.option("--loadDir", "loadDir", type=str, help="The directory to load the model from", required=True)
@click.option("--paramLoadName", "paramLoadName", type=str, help="File to load the model parameters from", required=True)
@click.option("--loadName", "loadName", type=str, help="Filename to load the model from", required=True)

# Other Parameters
@click.option("--device", "device", type=str, default="gpu", help="The device to train the model with (cpu or gpu)", required=False)
@click.option("--batchSize", "batchSize", type=int, default=0, help="The size of each minibatch of data (use 0 to use a single batch)", required=False)

# Bounding Box Filtering
@click.option("--removal_threshold", "removal_threshold", type=float, default=0.5, help="The threshold of predictions to remove if the confidence in that prediction is below this value", required=False)
@click.option("--score_thresh", "score_thresh", type=float, default=0.5, help="The score threshold to remove boxes in NMS. If the score is less than this value, remove it", required=False)
@click.option("--IoU_thresh", "IoU_thresh", type=float, default=0.1, help="The IoU threshold to update scores in NMS. If the IoU is greater than this value, update it's score", required=False)

# Focal Loss Function Hyperparameters
@click.option("--FL_alpha", "FL_alpha", type=float, default=4.0, help="The focal loss alpha parameter", required=False)
@click.option("--FL_gamma", "FL_gamma", type=float, default=2.0, help="The focal loss gamma parameter", required=False)
def predict(
    dataDir: str,
    loadDir: str,
    paramLoadName: str,
    loadName: str,
    
    device: Optional[str],
    batchSize: Optional[int],

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
    
    
    
    
    
    ### Data Loading ###
    
    # Ensure the directory exists
    assert os.path.isdir(loadDir), f"Load directory {loadDir} does not exist."
    
    # Ensure the parameter file exists
    assert os.path.isfile(os.path.join(loadDir, paramLoadName)), f"Load file {os.path.join(loadDir, paramLoadName)} does not exist."
    
    # Load in the model parameters
    with open(os.path.join(loadDir, paramLoadName), "r", encoding='utf-8') as f:
        data = json.load(f)
        
    # Store the loaded data as model parameters
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
        model = YOLOX(device, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_weight, category_Ids, removal_threshold, score_thresh, IoU_thresh)
        
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
    
    return 0




if __name__=='__main__':
    # Usage: python predict.py --dataDir=[dataDir] --loadDir=[loadDir] --paramLoadName=[paramLoadName] --loadName=[loadName]
    predict()
