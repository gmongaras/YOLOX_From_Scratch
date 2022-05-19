import torch
from math import sqrt
from random import uniform
import numpy as np
from PIL import Image





# Resize an image and its annotations
# Inputs:
#   img - An image in pytorch form in the shape (H, W, RGB)
#   ann - Dictionary of annotations for the image.
#   size - An iterable with the width and height to
#          resize the image to
# Outputs:
#   img - The resized image to the given size
#   ann - Resized/moved annotations
def resize(img, anns, size):
    # Convert the image to a Pillow object
    img = Image.fromarray(img)
    
    # Get the proportion to resize the image
    prop_w = size[0]/img.width
    prop_h = size[1]/img.height
    
    # Resize the image
    new_w = round(img.width*prop_w)
    new_h = round(img.height*prop_h)
    img = img.resize((new_w, new_h))
    
    # Convert the image back to a numpy array
    img = np.array(img, dtype=np.uint8).transpose(2,0,1)
    
    # Iterate over all annotations
    for a in range(0, len(anns["bbox"])):
        # Resize the bounding box by the proportions
        anns["bbox"][a][0] *= prop_w
        anns["bbox"][a][2] *= prop_w
        anns["bbox"][a][1] *= prop_h
        anns["bbox"][a][3] *= prop_h
    
    # Return the new image and annotations
    return img, anns


# Given 4 images with the same dimensions, and their
# annotations, create a mosaic of the images with the same dimenions
# Inputs:
#   imgs - A 4 item list with numpy arrays as images to augment.
#          The images can be different sizes.
#   anns - A 4 item list with dictionaries containing annotations
#          for each image respectively.
#          Each annotation has the following keys:
#          1. bbox - The bounding boxes in that image with four
#                    values each (x, y, w, h)
#          2. cls - The class of the objects in each bounding box
#          3. pix_cls - A tensor of the same dimensions as the
#                       original image where each pixel is the
#                       class in that pixel
# size - An iterable of size (height, width) which is the
#        size of the output image
# Output:
#   img - An image of the given size
#   new_anns - Combined annotations for the new image
def Mosaic(imgs, anns, size):
    ## The mosaic data augmentation takes 4 images of
    ## different sizes and follows the steps to create
    ## the final image:
    ## 1. Resize the images. In this case, I am resizing to
    ##    the shape of the output image
    ## 2. Combine all images into a single image where each
    ##    of the 4 resized images are a different corner.
    ## 3. Place the budning boxes to the correct areas
    ##    on the new image
    ## 4. Take a random cutout which is the size of the image
    ##    we want the final result to be. This cutout
    ##    can be anywhere on the combination of 4 images
    ## 5. Remove annotations that aren't in the cutout
    ## 6. Resize any annotations that are cutoff by the cutout
    
    
    # Ensure the images are in (H, W, RGB) format
    for i in range(0, len(imgs)):
        if imgs[i].shape[-1] != 3:
            imgs[i] = imgs[i].transpose(1, 2, 0)
    
    
    ## 1. Resize the images
    
    # Resize all images
    for img in range(0, len(imgs)):
        if imgs[img].shape[0] != size[0] or imgs[img].shape[1] != size[1]:
            imgs[img], anns[img] = resize(imgs[img], anns[img], size)
    
    
    
    
    ## 2. Combine the images
    
    # Create a tensor which is double the size of the
    # resized images
    img_combined = torch.zeros((3, size[0]*2, size[1]*2), requires_grad=False, dtype=torch.int16)
    
    # Fill the combined image with the 4 resized images
    img_combined[:, :size[0], :size[1]] = torch.tensor(imgs[0], dtype=torch.int16, requires_grad=False)
    img_combined[:, :size[0], size[1]:] = torch.tensor(imgs[1], dtype=torch.int16, requires_grad=False)
    img_combined[:, size[0]:, :size[1]] = torch.tensor(imgs[2], dtype=torch.int16, requires_grad=False)
    img_combined[:, size[0]:, size[1]:] = torch.tensor(imgs[3], dtype=torch.int16, requires_grad=False)
    
    
    ## 3. Move the bounding boxes
    
    # All bounding boxes and their class for
    # the combined image
    ann_bbox = []
    ann_cls = []
    
    # The first set of boudning boxes are already in the
    # correct location
    ann_bbox += anns[0]["bbox"]
    ann_cls += anns[0]["cls"]
    
    # The second set of boudning boxes need to be
    # moved to the right by size[0]
    ann_cls += anns[1]["cls"]
    for a in range(0, len(anns[1]["bbox"])):
        anns[1]["bbox"][a][0] += size[0]
        ann_bbox.append(anns[1]["bbox"][a])
        
    # The third set of boudning boxes need to be
    # moved down by size[1]
    ann_cls += anns[2]["cls"]
    for a in range(0, len(anns[2]["bbox"])):
        anns[2]["bbox"][a][1] += size[1]
        ann_bbox.append(anns[2]["bbox"][a])
        
    # The fourth set of boudning boxes need to be
    # moved to the right by size[0] and down by size[1]
    ann_cls += anns[3]["cls"]
    for a in range(0, len(anns[3]["bbox"])):
        anns[3]["bbox"][a][0] += size[0]
        anns[3]["bbox"][a][1] += size[1]
        ann_bbox.append(anns[3]["bbox"][a])
    
    # Make the annotations numpy arrays
    ann_bbox = np.array(ann_bbox, dtype=np.uint16)
    ann_cls = np.array(ann_cls, dtype=np.uint16)
    
    
    ## 4. Make a random cutout of the image
    
    # Get a random height and width from the image. Since
    # the image is twice the input size, we can get
    # a random value between root of the size and the size
    # to get a random (x, y) coordinate of the
    # top left of the cut. Then add the size
    # to get the bottom right of the cut
    
    topL_x = round(uniform(sqrt(size[0]), size[0]-sqrt(size[0])))
    topL_y = round(uniform(sqrt(size[1]), size[1]-sqrt(size[1])))
    
    # Get the final x,y coordinates
    cut_x = (topL_x, topL_x+size[0])
    cut_y = (topL_y, topL_y+size[0])
    
    # Get the random cut
    cutout = img_combined[:, cut_y[0]:cut_y[1], cut_x[0]:cut_x[1]]


    
    
    ## 5. Remove annotations not in the cutout
    
    # Get the (x, y) coordinates of the intersection
    xA = np.maximum(cut_x[0], ann_bbox[:, 0]).astype(np.int32)
    yA = np.maximum(cut_y[0], ann_bbox[:, 1]).astype(np.int32)
    xB = np.minimum(cut_x[1], ann_bbox[:, 0]+ann_bbox[:, 2]).astype(np.int32)
    yB = np.minimum(cut_y[1], ann_bbox[:, 1]+ann_bbox[:, 3]).astype(np.int32)
    
    # Get the intersection area between the cutout
    # and all bounding boxes
    intersectionArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    
    # Get all annotations inside the cutout (where
    # the intersection is not 0)
    ann_bbox = ann_bbox[intersectionArea != 0]
    ann_cls = ann_cls[intersectionArea != 0]
    
    
    ## 6. Resize the remaining bounding boxes
    
    # Move the top-left corner to be within the cutout
    mask = ann_bbox[:, 0] < cut_x[0]        # Where are the boxes that are below the image on the x-axis?
    ann_bbox[:, 2][mask] -= cut_x[0]-ann_bbox[:, 0][mask]   # Resize the bounding box to be the correct width
    ann_bbox[:, 0][mask] = 0         # Move the bounding box to be inside the cutout
    mask = ann_bbox[:, 1] < cut_y[0]        # Where are the boxes that are below the image on the y-axis?
    ann_bbox[:, 3][mask] -= cut_y[0]-ann_bbox[:, 1][mask]   # Resize the bounding box to be the correct width
    ann_bbox[:, 1][mask] = 0         # Move the bounding box to be inside the cutout
    
    # Move the rest of the boudning boxes
    ann_bbox[:, 0][ann_bbox[:, 0] >= cut_x[0]] -= cut_x[0]
    ann_bbox[:, 1][ann_bbox[:, 1] >= cut_y[0]] -= cut_y[0]
    
    # Move the width and height to be within the cutout
    mask = ann_bbox[:, 2]+ann_bbox[:, 0] > cut_x[1]-cut_x[0]
    ann_bbox[:, 2][mask] = cut_x[1]-cut_x[0]-ann_bbox[:, 0][mask]
    mask = ann_bbox[:, 3]+ann_bbox[:, 1] > cut_y[1]-cut_y[0]
    ann_bbox[:, 3][mask] = cut_y[1]-cut_y[0]-ann_bbox[:, 1][mask]
    
    
    
    # Create the pixel class map
    pix_cls = torch.zeros(cutout.shape[1:], dtype=torch.int16, requires_grad=False)
    for box_num in range(0, ann_bbox.shape[0]):
        box = ann_bbox[box_num]
        pix_cls[box[0]:box[0]+box[2], box[1]:box[1]+box[3]] = ann_cls[box_num]
    
    
    # Create the final annotation dictionary
    new_anns = {"bbox":ann_bbox.tolist(), "cls":ann_cls, "pix_cls":pix_cls}
    
    
    
    # Return the image and its annotations
    return cutout, new_anns






# Given two images, 
# Inputs:
#   img1/img2 - Images of different shapes to combine
#               as numpy arrays
#   labels1/labels2 - The labels for each of the input images
#   finalShape - The final width and height the resulting
#                image should be in
def mixup(img1, img2, labels1, labels2, finalShape, alpha=1.5):
    # Ensure the images are in (RGB, W, H) format
    if img1.shape[0] != 3:
        img1 = img1.transpose(2, 0, 1)
    if img2.shape[0] != 3:
        img2 = img2.transpose(2, 0, 1)
    
    # Get the max width and height out of the two images
    width = max(img1.shape[1], img2.shape[1])
    height = max(img1.shape[2], img2.shape[2])
    
    # Create an image of block pixels which is the
    # same shape as the max width and height
    new_img = np.zeros((3, width, height), dtype=np.float32)
    
    # Sample from a beta distribution to get the lambda value
    Lambda = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    
    # Use the formula by the paper to weight each image
    img1 = Lambda*img1
    img2 = (1-Lambda)*img2
    
    # Combine the images
    new_img[:, 0:img1.shape[1], 0:img1.shape[2]] = img1
    new_img[:, 0:img2.shape[1], 0:img2.shape[2]] += img2
    
    # Change the image to an integer image
    new_img = new_img.astype(np.int16)
    
    # Combine the annotations
    new_bbox = np.array(labels1["bbox"] + labels2["bbox"], dtype=np.int16)
    new_cls = labels1["cls"] + labels2["cls"]
    new_labels = {"bbox":new_bbox.tolist(), "cls":new_cls}
    
    # Resize the image to the given size
    new_img, new_labels = resize(new_img.transpose(1,2,0).astype(np.uint8), new_labels, finalShape)
    new_img = torch.tensor(new_img, dtype=torch.int16, requires_grad=False)
    
    # Add the pixel class to the labels
    pix_cls = torch.zeros(new_img.shape[1:], dtype=torch.int16, requires_grad=False)
    for box_num in range(0, len(new_bbox)):
        box = new_bbox[box_num]
        pix_cls[box[0]:box[0]+box[2], box[1]:box[1]+box[3]] = new_cls[box_num]
    new_labels["pix_cls"] = pix_cls
    
    return new_img, new_labels