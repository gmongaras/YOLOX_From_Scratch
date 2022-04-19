from YOLOX import YOLOX
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import random










def main():
    # Model Hyperparameters
    k = 10                  # The max number of annotations per image
    
    
    # COCO dataset parameters
    dataDir = "../coco"     # The location of the COCO dataset
    dataType = "val2017"    # The type of data being used in the COCO dataset
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    categories = ["person", "dog", "cat"]   # The categories to load in
    numToLoad = 10          # Max Number of data images to load in (use -1 for all)
    resize = 512            # Resize the images to a quare pixel value (can be 1024, 512, or 256)
    
    
    
    
    ### Data Loading
    
    # Initialize the COCO data
    coco=COCO(annFile)
    
    # Get the image data and annotations
    img_data = []
    ann_data = dict()
    seen = []
    for c in categories:
        catIds = coco.getCatIds(catNms=[c])
        imgIds = coco.getImgIds(catIds=catIds)
        sub_img_data = coco.loadImgs(imgIds)
        
        notSeen = []
        for i in sub_img_data:
            if i['id'] not in seen:
                seen.append(i['id'])
                notSeen.append(i)
        img_data += notSeen
        
        for i in notSeen:
            ann_data[i['id']] = coco.loadAnns(coco.getAnnIds(imgIds=i['id'], iscrowd=None))
    
    # Get a subset if requested
    if numToLoad > 0:
        img_data = img_data[:numToLoad]
    
    
    # Load in the actual images
    imgs = []
    print("\nLoading images...")
    for img_d in img_data:
        img = io.imread(img_d['coco_url'])
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = img.resize((resize, resize))
        imgs.append(np.array(img))
    imgs = np.array(imgs)
    print("Images Loaded")
    
    
    # Turn the annotations into a better form
    anns = []
    for img_d in img_data:
        ann = ann_data[img_d["id"]][:k]
        ann_bbox = []
        for a in ann:
            ann_bbox.append(a["bbox"])
        anns.append(ann_bbox)
    
    
    
    
    ### Model Training
    model = YOLOX()
    
    
    
    
    #### Model Testing
    print()




if __name__=='__main__':
    main()