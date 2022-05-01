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




device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
cpu = torch.device('cpu')







def main():
    # Model Hyperparameters
    k = 10                  # The max number of annotations per image
    numEpochs = 300         # The number of epochs to train the model for
    batchSize = 128         # The size of each minibatch
    warmupEpochs = 5        # Number of warmup epochs to train the model for
    lr_init = 0.01          # Initial learning rate
    weightDecay = 0.0005    # Amount to decay weights over time
    momentum = 0.9          # Momentum of the SGD optimizer
    SPPDim = 256            # The height and width dimension to convert FPN 
                            # (Feature Pyramid Network) encodings to
    numCats = 3             # The number of categories to predict from
    reg_consts = (          # The contraints on the regression size
        0, 64, 128, 256     # Basically constraints on how large the bounding
        )                   # boxes can be for each level in the network
    
    # Loss Function Hyperparameters
    FL_alpha = 4            # The focal loss alpha parameter
    FL_gamma = 2            # The focal loss gamma paramter
    reg_weight = 0.5        # Percent to weight regression loss over other loss
    
    
    # COCO dataset parameters
    dataDir = "../coco"     # The location of the COCO dataset
    dataType = "val2017"    # The type of data being used in the COCO dataset
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    categories = ["person", "dog", "cat"]   # The categories to load in
    numToLoad = 10           # Max Number of data images to load in (use -1 for all)
    resize = 256            # Resize the images to a quare pixel value (can be 1024, 512, or 256)
    
    
    # Ensure the number of categories is equal to the list
    # of categories
    assert numCats == len(categories), "Number of categories (numCats) must be equal to the length of the categories list (categories)"
    
    
    
    ### Data Loading
    
    # Initialize the COCO data
    coco=COCO(annFile)
    
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
        prop = resize/(max(img.height, img.width)) # proportion to resize image
        new_h = round(img.height*prop)
        new_w = round(img.width*prop)
        img = img.resize((new_w, new_h))
        #img.thumbnail((new_h, new_w), Image.ANTIALIAS) # Resize the image
        img = np.array(img)
        
        # Pad with zeros if needed
        pad = [(resize-img.shape[0])-(resize-img.shape[0])//2, (resize-img.shape[1])-(resize-img.shape[1])//2]
        img = np.pad(img, (((resize-img.shape[0])-(resize-img.shape[0])//2, ((resize-img.shape[0])//2)), ((resize-img.shape[1])-(resize-img.shape[1])//2, ((resize-img.shape[1])//2)), (0, 0)), mode='constant')
        
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
    seq_category_Ids = {i:category_Ids[i] for i in category_Ids.keys()}
    seq_category_Ids["Background"] = 0
    seq_category_Ids_rev = {seq_category_Ids[i]:i for i in seq_category_Ids.keys()}
    
    
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
    
    
    
    
    ### Model Training
    torch.autograd.set_detect_anomaly(True)
    model = YOLOX(device, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, SPPDim, numCats, FL_alpha, FL_gamma, reg_consts, reg_weight)
    model.train(imgs, anns)
    
    
    
    
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
    main()