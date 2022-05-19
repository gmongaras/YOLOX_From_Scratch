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
        return -self.FL_alpha*((1-p_t)**self.FL_gamma)*torch.log(p_t)
    
    
    # Get the binary cross entropy loss given class predictions
    # Inputs:
    #   y_hat - A tensor of shape (batchSize, H, W) where each pixel is
    #           the class prediction for that pixel
    #   y - A tensor of shape (batchSize, H, W) where each pixel
    #       is the class for that pixel
    # Outputs:
    #   The Binary Cross Entropy of the input tensor
    def BinaryCrossEntropy(self, y_hat, y):
        return torch.nn.BCEWithLogitsLoss(reduction='sum')(y_hat, y)
    
    
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
        y_hat = torch.where(y_hat > 0.999999, y_hat-0.000001, y_hat)
        
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
    #   lossVals - A tensor where each value is the loss for that
    #               image (numImages)
    #   GIoU - A tensor where each value is the GIoU value for that
    #               image (numImages). Note, this is not the loss value
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
        
        # The intersection is 0 if the area is 0 or below or
        # the positive area of the intersection if it's greater than 0
        I = torch.where(torch.logical_and(x_2_I > x_1_I, y_2_I > y_1_I), (x_2_I - x_1_I) * (y_2_I - y_1_I), torch.tensor(0.0))
        nn.functional.relu(I, inplace=True)
        
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
        IoU[U == 0] = 0
        
        # 8: Calculate the GIoU
        GIoU = IoU - ((A_c - U) / A_c)
        
        # Ensure the GIoU value is between -1 and 1
        assert torch.where(GIoU > 1)[0].shape[0] == 0, "GIoU is greater than 1... This shouldn't happen"
        assert torch.where(GIoU < -1)[0].shape[0] == 0, "GIoU is less than -1... This shouldn't happen"
        
        # 9: Save the values as loss
        # (we use 1 - GIoU as we want to minimize the GIoU)
        lossVals = 1 - GIoU
        
        # Create a tensor of the losses and return it
        return lossVals, GIoU

    
    
    # Get the IoU loss given two bounding boxes
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
    #   IoU_loss - A tensor where each value is the loss for that
    #               image (numImages)
    #   IoU - A tensor where each value is the IoU value for that
    #               image (numImages). Note, this is not the loss value
    def IoU(self, pred, GT):
        # Ensure the shapes are the same
        if len(pred.shape) > len(GT.shape):
            GT = torch.unsqueeze(GT, dim=0)
        elif len(pred.shape) < len(GT.shape):
            pred = torch.unsqueeze(pred, dim=0)
        
        # Get the (x, y) coordinates of the intersection
        xA = torch.maximum(pred[:, 0], GT[:, 0])
        yA = torch.maximum(pred[:, 1], GT[:, 1])
        xB = torch.minimum(pred[:, 0]+pred[:, 2], GT[:, 0]+GT[:, 2])
        yB = torch.minimum(pred[:, 1]+pred[:, 3], GT[:, 1]+GT[:, 3])
        
        # Get the area of the intersection
        intersectionArea = torch.maximum(torch.tensor(0), xB - xA + 1) * torch.maximum(torch.tensor(0), yB - yA + 1)
        
        # Compute the area of both rectangles
        areaA = (pred[:, 2]+1)*(pred[:, 3]+1)
        areaB = (GT[:, 2]+1)*(GT[:, 3]+1)
        
        # Get the union of the rectangles
        union = areaA + areaB - intersectionArea
        
        # Compute the intersection over union
        IoU = intersectionArea/union
        IoU[union == 0] = 0
        
        # Get the IoU loss
        IoU_loss = 1-IoU
        
        return IoU_loss, IoU