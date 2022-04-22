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
    def __init__(self, device, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, SPPDim, numCats, FL_alpha, FL_gamma):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.k = k
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        self.device=device
        
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
            for i in range(0, len(X_batches)):
                # Load the batch data onto the gpu if available
                X_b = X_batches[i].to(self.device)
                y_b = y_batches[i]
                
                # Get a prediction of the bounding boxes on the input image
                cls, reg, iou = self.forward(X_b)
                
                # Cumulate the loss across the three predictions
                totalLoss = 0
                
                # Iterate over the three sets of predictions
                for p in range(0, 3):
                    # Get the current set of predictions
                    cls_p = cls[p].permute(0, 2, 3, 1)
                    reg_p = reg[p].permute(0, 2, 3, 1)
                    iou_p = iou[p].permute(0, 2, 3, 1)
                    
                    
                    
                    ### Class predictions
                    
                    # Send the data through the Focal Loss function
                    FL = torch.mean((1/cls_p.shape[0])*torch.sum(self.losses.FocalLoss(cls_p.to(cpu), torch.stack([i["pix_cls"] for i in y_b]).to(cpu)), dim=-1))
                    
                    
                    
                    
                    ### Regression predictions
                    ##
                    
                    # Use GIoU for loss
                    
                    
                    
                    
                    ### IoU (object) predictions
                    
                    #iouLoss = 
                    
                    # Convert to centered values (In Multipositives section)
                    # Look at FCOS
                    
                    # Use BCE on sigmoid outputs for loss
                    
                    # https://medium.com/swlh/fcos-walkthrough-the-fully-convolutional-approach-to-object-detection-777f614268c
                    
                    
                    
                    
                    # Get the final loss for this prediction
                    finalLoss = FL# + regLoss + iouLoss
                    totalLoss += finalLoss
                
                
                ### Updating the model
                if epoch%5 == 0:
                    plt.imshow(torch.argmax(cls_p[0], dim=-1).cpu().detach().numpy(), interpolation='nearest')
                    plt.show()
                #plt.imshow(torch.argmax(y_b[0]["pix_cls"], dim=-1).cpu().detach().numpy(), interpolation='nearest')
                #plt.show()
                #plt.imshow(torch.argmax(cls_p[0], dim=-1).cpu().detach().numpy(), interpolation='nearest')
                #plt.show()
                
                # Backpropogate the loss
                totalLoss.backward()
                
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