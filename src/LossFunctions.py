import torch
from torch import nn


# Class used to store loss functions needed for the
# YOLOX model
class LossFunctions():
    # Initialize the loss functions
    # Inputs:
    #   ;
    def __init__(self, FL_gamma, FL_alpha):
        # Focal Loss Parameters
        self.FL_gamma = FL_gamma
        self.FL_alpha = FL_alpha
        
    
    # Get the focal loss given class predictions
    # Inputs:
    #   z - A tensor of shape (batchSize, H, W, classes) where each channel is
    #       all class predictions for that pixel
    #   y - A tensor of shape (batchSize, H, W, classes) where each pixel
    #       is a one-hot encoded matrix where the size is equal to the number
    #       of classes and the 1 is located in the correct class
    # Outputs:
    #   The focal loss of the inputs tensor
    def FocalLoss(self, z, y):
        # Convert the inputs into a p matrix as the formula suggests
        p = nn.Sigmoid()(z)
        
        # Get the p_t value using the labels
        p_t = torch.where(y==1, p,  1-p)
        
        # Ensure no Nan values
        p_t = torch.where(p_t < 0.000001, p_t+0.000001, p_t)
        p_t = torch.where(p_t > 0.999999, p_t-0.000001, p_t)
        
        # Return the loss value
        return -self.FL_alpha*((1-p_t)**self.FL_gamma)*torch.log(p_t)
        
        # Compute the loss and return it
        return -((y+1)/2)*self.FL_alpha*((1-p)**self.FL_gamma)*torch.log(p) - \
                ((1-y)/2)*((1-self.FL_alpha)*(p**self.FL_gamma))*torch.log(1-p)