import torch
from torch import nn


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
        # If the input is an empty tensor, return 0
        if y_hat.size() == torch.Size([0]) or y.size() == torch.Size([0]):
            return 0
        
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
    
    
    # Get the cross entropy loss given class predictions
    # Inputs:
    #   y_hat - A tensor of shape (batchSize, H, W, 4) where each set of 4 is
    #           a predicted bounding box
    #   y - A tensor of shape (batchSize, H, W, 4) where each set of 4
    #       is a ground truth bounding box
    # Outputs:
    #   The Cross Entropy of the input tensor
    def CrossEntropy(self, y_hat, y):
        # Ensure no Nan values
        y_hat = torch.where(y_hat < 0.000001, y_hat+0.000001, y_hat)
        #y_hat = torch.where(y_hat > 0.999999, y_hat-0.000001, y_hat)
        
        # Return the loss value
        return -torch.mean(y*torch.log(y_hat))


    # Get the Generic IoU loss given two bounding boxes
    # Inputs:
    #   pred - A set of predicted bounding boxes with 4 elements:
    #     1. top-left x coordinate of the bounding box
    #     2. top-left y coordinate of the bounding box
    #     3. heihgt of the bounding box
    #     4. width of the bounding box
    #   Y - A set of ground truth boudning boxes with the same shape
    #       as the predicted bounding boxes
    #   The tensors have shape: (numImages, 4)
    # Outputs:
    #   A tensor where each value is the loss for that
    #   image (numImages)
    def GIoU(self, pred, GT):
        # If either the predictions or ground truth values
        # are empty, return 0
        if pred.shape[0] == 0 or GT.shape[0] == 0:
            return 0
        
        # Convert the boxes to the correct form:
        #   1: x_1 - The lower/left-most x value
        #   2: y_1 - The lower/upper-most y value
        #   3: x_2 - The higher/right-most x value
        #   4: y_2 - The higher/bottom-most y value
        B_p_stack = torch.stack([torch.stack([X[0], X[1], X[0]+X[2], X[1]+X[3]]) for X in pred]).float()
        B_g_stack = torch.stack([torch.stack([Y[0], Y[1], Y[0]+Y[2], Y[1]+Y[3]]) for Y in GT]).float()
        
        # 1: Store the predictions in separate variables
        # and ensure x_2 > x_1 and y_2 > y_1
        if B_p_stack.shape[-1] != B_g_stack.shape[-1]:
            B_p_stack = B_p_stack.T
        x_1_p = torch.minimum(B_p_stack[:, 0], B_p_stack[:, 2])
        x_2_p = torch.maximum(B_p_stack[:, 0], B_p_stack[:, 2])
        y_1_p = torch.minimum(B_p_stack[:, 1], B_p_stack[:, 3])
        y_2_p = torch.maximum(B_p_stack[:, 1], B_p_stack[:, 3])
        
        # Array to save all loss values
        lossVals = []
        
        # Store the value in B_g in separate variables
        x_1_g = B_g_stack[:, 0]
        x_2_g = B_g_stack[:, 2]
        y_1_g = B_g_stack[:, 1]
        y_2_g = B_g_stack[:, 3]
        
        # 2: Calculate area of B_g
        A_g = (x_2_g - x_1_g) * (y_2_g - y_1_g)
        
        # 3: Calculate area of B_p
        A_p = (x_2_p - x_1_p) * (y_2_p - y_1_p)
        
        # 4: Calculate intersection between B_p and B_g
        x_1_I = torch.maximum(x_1_p, x_1_g)
        x_2_I = torch.minimum(x_2_p, x_2_g)
        y_1_I = torch.maximum(y_1_p, y_1_g)
        y_2_I = torch.minimum(y_2_p, y_2_g)
        I = torch.zeros(x_1_I.shape)
        I[torch.any(torch.logical_or(x_2_I > x_1_I, y_2_I > y_1_I))] = \
            (x_2_I - x_1_I) * (y_2_I - y_1_I)
        
        # 5: Find coordinate of smallest enclosing box B_c
        x_1_c = torch.minimum(x_1_p, x_1_g)
        x_2_c = torch.maximum(x_2_p, x_2_g)
        y_1_c = torch.minimum(y_1_p, y_1_g)
        y_2_c = torch.maximum(y_2_p, y_2_g)
        
        # 6: Calculate area of B_c
        A_c = (x_2_c - x_1_c) * (y_2_c - y_1_c)
        
        # 7: Calculate IoU
        U = A_p + A_g - I
        IoU = I/U
        
        # 8: Calculate the GIoU
        GIoU = IoU - ((A_c - U) / A_c)
        
        # 9: Save the values as loss
        # (we use 1 - GIoI as we want to minimize the GIoU)
        # (we also take the absolute value so it's not negative)
        lossVals = 1 - GIoU
        
        # Create a tensor of the losses and return it
        return lossVals