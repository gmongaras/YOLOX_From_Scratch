from Darknet53 import Darknet53
from LossFunctions import LossFunctions

import torch
from torch import nn
import numpy as np


cpu = torch.device('cpu')




class YOLOX(nn.Module):
    # Initialze the model
    # Inputs:
    #   device - Device to put the network on
    #   k - The number of possible bounding boxes
    #   numEpochs - The number of epochs to train the model for
    #   batchSize - The size of each minibatch
    #   warmupEpochs - Number of warmup epochs to train the model for
    #   lr_init - Initial learning rate
    #   weightDecay - Amount to decay weights over time
    #   momentum - Momentum of the SGD optimizer
    #   SPPDim - The height and width dimension to convert FPN 
    #            (Feature Pyramid Network) encodings to
    #   numCats - The number of categories to predict from
    #   FL_alpha - The focal loss alpha parameter
    #   FL_gamma - The focal loss gamma parameter
    #   reg_consts - The regression constraints (should be 4 values)
    def __init__(self, device, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, SPPDim, numCats, FL_alpha, FL_gamma, reg_consts):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.k = k
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        self.device = device
        assert len(reg_consts) == 4, "The regression constrainsts should contain 4 values"
        self.reg_consts = reg_consts
        self.numCats = numCats
        
        # The darknet backbone
        self.darknet = Darknet53(device, SPPDim)
        
        # The loss functions
        self.losses = LossFunctions(FL_gamma, FL_alpha)
        
        # Use 1x1 convolutions so that each value will have
        # the exact same dimensions
        self.conv11_1 = nn.Conv2d(1024, 256, kernel_size=1, device=device)
        self.conv11_2 = nn.Conv2d(512, 256, kernel_size=1, device=device)
        self.conv11_3 = nn.Conv2d(256, 256, kernel_size=1, device=device)
        
        # The class convolution
        self.clsConv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, numCats+1, kernel_size=1)
        ).to(device)
        
        # The regression convolution
        self.regConv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        ).to(device)
        
        # The regression 1x1
        self.reg11 = nn.Conv2d(256, 4, kernel_size=1, device=device)
        
        # The IoU 1x1
        self.iou11 = nn.Conv2d(256, 1, kernel_size=1, device=device)
        
        # Create the optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr_init*batchSize/64, momentum=momentum, weight_decay=weightDecay)
    
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=numEpochs)
    
    
    # Get a prediction for the bounding box given some images
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    def forward(self, X):
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Send the inputs through the Darknet backbone
        FPN1, FPN2, FPN3 = self.darknet(X)
        
        # Send each input through a 1x1 conv to get all FPN
        # values with the same exact number of channels
        v1 = self.conv11_1(FPN1)
        v2 = self.conv11_2(FPN2)
        v3 = self.conv11_3(FPN3)
        
        # Send the inputs through the class convolution
        clsConv1 = self.clsConv(v1)
        clsConv2 = self.clsConv(v2)
        clsConv3 = self.clsConv(v3)
        
        # Send the inputs through the regression convolution
        regConv1 = self.regConv(v1)
        regConv2 = self.regConv(v2)
        regConv3 = self.regConv(v3)
        
        # Send the regression covolution outputs through the
        # regression 1x1 and the IoU 1x1
        reg11_1 = self.reg11(regConv1)
        reg11_2 = self.reg11(regConv2)
        reg11_3 = self.reg11(regConv3)
        iou11_1 = self.iou11(regConv1)
        iou11_2 = self.iou11(regConv2)
        iou11_3 = self.iou11(regConv3)
        
        # Send the IoU through a sigmoid function to get a value
        # between 0 and 1.
        iou11_1 = torch.sigmoid(iou11_1)
        iou11_2 = torch.sigmoid(iou11_2)
        iou11_3 = torch.sigmoid(iou11_3)
        
        # Send the reg through a ReLU function to get a value
        # above 0
        reg11_1 = torch.relu(reg11_1)
        reg11_2 = torch.relu(reg11_2)
        reg11_3 = torch.relu(reg11_3)
        
        # Return the data as arrays
        return [clsConv1,clsConv2,clsConv3], [reg11_1,reg11_2,reg11_3], [iou11_1,iou11_2,iou11_3]
    
    
    # Train the network on training data
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    #   y - The labels for each input (correct bounding boxes to place on image)
    def train(self, X, y):
        from matplotlib import pyplot as plt
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=cpu, requires_grad=False)
        
        # Update the models `numEpochs` number of times
        for epoch in range(1, self.numEpochs+1):
            # Get a randomized set of indices
            idx = torch.randperm(X.shape[0], device=cpu)
            
            # Randomly split the data into batches
            X_batches = torch.split(X[idx], self.batchSize)
            y_temp = np.array(y, dtype=object)[idx.cpu().detach().numpy()].tolist()
            y_batches = [y_temp[self.batchSize*i:self.batchSize*(i+1)] for i in range(0, (X.shape[0]//self.batchSize))]
            if y_batches == []:
                y_batches = [y]
            else:
                y_batches += y_temp[self.batchSize*(X.shape[0]//self.batchSize):]
                if type(y_batches[-1]) == dict:
                    y_batches[-1] = [y_batches[-1]]
            
            # The loss over all batches
            batchLoss = 0
            
            # Iterate over all batches
            for batch in range(0, len(X_batches)):
                # Load the batch data onto the gpu if available
                X_b = X_batches[batch].to(self.device)
                y_b = y_batches[batch]
                
                # Get a prediction of the bounding boxes on the input image
                cls, reg, iou = self.forward(X_b)
                
                # Cumulate the loss across the three predictions
                totalLoss = 0
                
                # Iterate over the three sets of predictions
                for p in range(0, 3):
                    # Get the current set of predictions in a flattened state
                    cls_p = cls[p].permute(0, 2, 3, 1)
                    cls_p = cls_p.reshape(cls_p.shape[0], cls_p.shape[1]*cls_p.shape[2], cls_p.shape[3])
                    reg_p = reg[p].permute(0, 2, 3, 1)
                    reg_p = reg_p.reshape(reg_p.shape[0], reg_p.shape[1]*reg_p.shape[2], reg_p.shape[3])
                    iou_p = iou[p].permute(0, 2, 3, 1)
                    iou_p = iou_p.reshape(iou_p.shape[0], iou_p.shape[1]*iou_p.shape[2], iou_p.shape[3])
                    
                    
                    ### Positive Filtering
                    
                    # First we nee dto filter the predictions so that the
                    # good predictions are kept
                    
                    # The regression constraints for this level
                    reg_const_low = self.reg_consts[p]
                    reg_const_high = self.reg_consts[p+1]
                    
                    # Create a matrix of positive/negative targets. A positive target
                    # is one which is within a certain threshold. So, the width and
                    # height should be within a predefined threshold.
                    reg_labels = torch.where(torch.logical_and(torch.max(reg_p[:,:,2:], dim=-1)[0] < reg_const_high, torch.min(reg_p[:,:,2:], dim=-1)[0] > reg_const_low), 1, 0)
                    
                    
                    ### GIoU Loss calculation
                    
                    
                    # Each prediction finds the boundary box with the best IoU
                    # which we will save to be used in the other functions.
                    
                    # Ensure no negative values
                    #bbox = torch.abs(reg_p[b_num])
                    #bbox[torch.any(bbox > X.shape[-1])] = X.shape[-1]
                    bbox = X.shape[-1]/(1+torch.exp(-reg_p[batch]))
                    #bbox = reg_p[b_num, reg_labels[b_num] == 1]
                    
                    # Get the best bounding box ground truth to the
                    # prediction and the GIoU of that bounding box
                    GIoULosses = self.losses.GIoU(bbox, torch.tensor(y_b[batch]['bbox'], dtype=float, requires_grad=False))
                    
                    # Get the indices and values for the best 
                    # loss for each bounding box prediction
                    # (basically what ground state bounding box
                    # is closest to each predicted one?)
                    vals, minIdx = torch.min(GIoULosses, dim=-1)
                    
                    # Get the min loss for each bounding
                    # box prediction and sum it up
                    GIoU_loss = torch.sum(vals)
                    
                    
                    
                    
                    ### Class predictions
                    
                    # Get the ground truth value for each prediction as a one-hot vector
                    GT = torch.zeros(list(cls_p.shape[:-1])+[self.numCats+1])
                    for p_num in range(cls_p[batch].shape[0]): # Iterate over all predictions
                        # Label as background if not a positive pred
                        if reg_labels[batch, p_num] == 0:
                            GT[batch, p_num, 0] = 1
                            continue
                        
                        # Store the GT value as a 1 in the one-hot vector
                        GT[batch, p_num][y_b[batch]['pix_cls'][torch.div(reg_p[batch, p_num, 2].int(), 2, rounding_mode="trunc"), torch.div(reg_p[batch, p_num, 2].int(), 2, rounding_mode="trunc")]] = 1
                    
                    # Get the image class ground truth values
                    #y_b_cls = torch.stack([i["pix_cls"] for i in y_b]).to(cpu)
                    
                    # Resize the ground truth value to be the same shape as the
                    # predicted values
                    #y_b_cls = torch.nn.functional.interpolate(y_b_cls.reshape(y_b_cls.shape[0], 1, y_b_cls.shape[1], y_b_cls.shape[2]).float(), cls_p.shape[1]).int().squeeze()
                    
                    # One hot encode the ground truth values
                    #y_b_cls = torch.nn.functional.one_hot(torch.tensor(y_b_cls, dtype=int, device=cpu), self.numCats+1)
                    
                    # Send the data through the Focal Loss function
                    FL = self.losses.FocalLoss(cls_p.to(cpu), GT)
                    
                    
                    
                    
                    ### Regression predictions
                    
                    # Get the centerness of all bounding box predictions
                    # Notes: Centerness defined in the FCOS paper
                    #reg_cent = 1
                    
                    # Use GIoU for loss
                    
                    
                    
                    
                    ### IoU (objectiveness) predictions
                    
                    # Compute the IoU of the predictions
                    
                    ## Compute IoU, then use BCE
                    # - for IoU, compare each bounding box with their ground truth BB
                    # - IoU loss = 1-IoU
                    
                    #iouLoss = self.losses.BinaryCrossEntropy(iou_p.to(cpu), torch.stack([i["pix_obj"] for i in y_b]).to(cpu))
                    
                    # Convert to centered values (In Multipositives section)
                    # Look at FCOS
                    
                    # Use BCE on sigmoid outputs for loss
                    
                    # https://medium.com/swlh/fcos-walkthrough-the-fully-convolutional-approach-to-object-detection-777f614268c
                    
                    
                    
                    
                    # Get the final loss for this prediction
                    #finalLoss = FL# + regLoss + iouLoss
                    finalLoss = GIoU_loss
                    totalLoss += finalLoss
                
                
                ### Updating the model
                #if epoch%5 == 0:
                #    plt.imshow(torch.argmax(cls_p[0], dim=-1).cpu().detach().numpy(), interpolation='nearest')
                #    plt.show()
                #if epoch%5 == 0:
                #    plt.imshow(iou_p[0].cpu().detach().numpy(), interpolation='nearest')
                #    plt.show()
                
                # Backpropogate the loss
                totalLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                
                # Step the optimizer
                self.optimizer.step()
                
                # Step the learning rate optimizer after the warmup steps
                if epoch > self.warmupEpochs:
                    self.scheduler.step()
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Update the batch loss
                batchLoss += totalLoss.cpu().detach().numpy().item()
                
            print(f"Step #{epoch}      Total Batch Loss: {batchLoss}")
        
        return 0
        
        
    def predict(self):
        #https://medium.com/swlh/fcos-walkthrough-the-fully-convolutional-approach-to-object-detection-777f614268c
        # Look in inference mode
        return 1