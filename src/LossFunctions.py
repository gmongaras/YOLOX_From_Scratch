import torch
from torch import nn
from LossHelper import generalized_box_iou


# Class used to store loss functions needed for the
# YOLOX model
class LossFunctions():
    # Initialize the loss functions
    # Inputs:
    #   FL_gamma - The gamma parameter for the Focal Loss
    #   FL_alpha - The alpha parameter for the Focal Loss
    def __init__(self, FL_gamma, FL_alpha):
        # Focal Loss Parameters
        self.FL_gamma = FL_gamma
        self.FL_alpha = FL_alpha
        
    
    # Get the focal loss given class predictions
    # Inputs:
    #   z - A tensor of shape (batchSize, H, W, classes) where each channel is
    #       all class predictions for that pixel
    #   y - A tensor of shape (batchSize, H, W, classes) where each channel
    #       is a one-hot encoded matrix where the size is equal to the number
    #       of classes and the 1 is located in the correct class
    # Outputs:
    #   The focal loss of the inputs tensor
    def FocalLoss(self, z, y):
        # Convert the inputs into a p matrix as the formula suggests
        p = nn.Sigmoid()(z)
        
        # Get the p_t value using the labels
        p_t = torch.where(y == 1, p,  1-p)
        
        # Ensure no Nan values
        p_t = torch.where(p_t < 0.000001, p_t+0.000001, p_t)
        p_t = torch.where(p_t > 0.999999, p_t-0.000001, p_t)
        
        # Return the loss value
        return torch.mean((1/z.shape[0])*torch.sum(
            -self.FL_alpha*((1-p_t)**self.FL_gamma)*torch.log(p_t)
            , dim=-1))
        
        # Compute the loss and return it
        return -((y+1)/2)*self.FL_alpha*((1-p)**self.FL_gamma)*torch.log(p) - \
                ((1-y)/2)*((1-self.FL_alpha)*(p**self.FL_gamma))*torch.log(1-p)
    
    
    # Get the binary cross entropy loss given class predictions
    # Inputs:
    #   y_hat - A tensor of shape (batchSize, H, W) where each pixel is
    #           the class prediction for that pixel
    #   y - A tensor of shape (batchSize, H, W) where each pixel
    #       is the class for that pixel
    # Outputs:
    #   The Binary Cross Entropy of the input tensor
    def BinaryCrossEntropy(self, y_hat, y):
        # Flatten the tensors if the dimensions are over two
        if len(y_hat.shape) > 2:
            y_hat = torch.flatten(y_hat, 1, -1)
        if len(y.shape) > 2:
            y = torch.flatten(y, 1, -1)
        
        # Ensure no Nan values
        y_hat = torch.where(y_hat < 0.000001, y_hat+0.000001, y_hat)
        y_hat = torch.where(y_hat > 0.999999, y_hat-0.000001, y_hat)
        
        # Return the loss value
        return -torch.mean(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat))

    
    # Thank to Adrian Rosebrock for supplying code on
    # PyImageSearch. My code is an editted form of theirs. You
    # could say, I really o U Hahahaha!
    # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    
    # This function returns the IoU loss given two bounding boxes
    # Inputs:
    #   X and Y - Bounding boxes with 4 elements:
    #   1. top-left x coordinate of the bounding box
    #   2. top-left y coordinate of the bounding box
    #   3. heihgt of the bounding box
    #   4. width of the bounding box
    def IoU(self, X, Y):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(X[0], Y[0])
        yA = max(X[1], Y[1])
        xB = min(X[0]+X[2], Y[0]+Y[2])
        yB = min(X[0]+X[3], Y[0]+Y[3])
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = ((X[0]+X[2]) - X[0] + 1) * ((X[1]+X[3]) - X[1] + 1)
        boxBArea = ((Y[0]+Y[2]) - Y[0] + 1) * ((Y[1]+Y[3]) - Y[1] + 1)
        
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        union = float(boxAArea + boxBArea - interArea)
        iou = interArea / union
        
        # return the intersection over union value and the
        # union value
        return iou, union


    # Thanks to jw9730 for supplying this code on GitHub.
    # I guess you could also say, I o U too. :)
    # https://github.com/jw9730/ori-giou/commits?author=jw9730
    # Get the Generic IoU loss given two bounding boxes
    # Inputs:
    #   X and Y - Bounding boxes with 4 elements:
    #   1. top-left x coordinate of the bounding box
    #   2. top-left y coordinate of the bounding box
    #   3. heihgt of the bounding box
    #   4. width of the bounding box
    def GIoU(self, box1, box2):
        # Convert the boxes to the correct form
        box1 = torch.stack([torch.stack([X[0], X[1], X[0], X[1]+X[3], X[0]+X[2], X[1]+X[3], X[0]+X[2], X[1]]) for X in box1])
        box2 = torch.stack([torch.stack([Y[0], Y[1], Y[0], Y[1]+Y[3], Y[0]+Y[2], Y[1]+Y[3], Y[0]+Y[2], Y[1]]) for Y in box2])
        
        return generalized_box_iou(box1.float(), box2.float())