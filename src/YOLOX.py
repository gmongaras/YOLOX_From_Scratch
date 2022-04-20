import torch
from torch import nn
import numpy as np


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')






class Darknet53(nn.Module):
    # Initialize the mode
    # Inputs:
    #   SPPDim - The dimension to convert each image to (output image will be (C x SPPDim x SPPDim))
    #   inChanSize - The number of input channels (3 for RGB)
    def __init__(self, SPPDim, inChanSize=3):
        super(Darknet53, self).__init__()
        
        ### Darknet blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(inChanSize, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
        ).to(device)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block4 = [nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        ).to(device) for i in range(0, 2)]
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block6 = [nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        ).to(device) for i in range(0, 8)]
        self.block7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block8 = [nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
        ).to(device) for i in range(0, 8)]
        self.block9 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block10 = [nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        ).to(device) for i in range(0, 4)]
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.connected = nn.Linear(1024, 1000)
        
        ### Output/prediction blocks
        self.conv11_1 = nn.Conv2d(1024, 512, 1)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=1),
        ).to(device=device)
        
        self.conv11_2 = nn.Conv2d(512, 256, 1)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
        ).to(device=device)
        
        self.SPP = torch.nn.AdaptiveMaxPool2d(SPPDim)
        
    
    # Get a prediction from the model
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    # Outputs:
    #   p1, p2, p3 - Predictions from different levels in the model 
    def forward(self, X):
        ### Send the inputs through the darknet backbone
        
        v = self.block1(X)
        
        res = torch.clone(v)
        v = self.block2(v)+res
        v = self.block3(v)
        
        res = torch.clone(v)
        v = self.block4[0](v)+res
        res = torch.clone(v)
        v = self.block4[1](v)+res
        
        v = self.block5(v)
        
        for i in range(0, 8):
            res = torch.clone(v)
            v = self.block6[i](v)+res
            
        ck1 = torch.clone(v)
        
        v = self.block7(v)
        
        for i in range(0, 8):
            res = torch.clone(v)
            v = self.block8[i](v)+res
        
        ck2 = torch.clone(v)
        
        v = self.block9(v)
        
        for i in range(0, 4):
            res = torch.clone(v)
            v = self.block10[i](v)+res
        del res
        
        ck3 = torch.clone(v)
        
        v = torch.squeeze(self.globalPool(v))
        
        v = self.connected(v)
    
        ### Now send the three checkpoints through
        ### convolutions and upsampling to get three predictions
        
        # The first prediction is the last checkpoint
        p1 = torch.clone(ck3)
        
        # Second prediction
        v = self.conv11_1(ck3)
        v = self.upsample_1(v)
        v = torch.cat((v, ck2), dim=1)
        v = self.conv_1(v)
        p2 = torch.clone(v)
        
        # Third prediction
        v = self.conv11_2(p2)
        v = self.upsample_2(v)
        v = torch.cat((v, ck1), dim=1)
        v = self.conv_2(v)
        p3 = v
        
        # Apply Spatial Pyramid Pooling (SPP) to each prediction
        # so that the height and width are all equal
        p1 = self.SPP(p1)
        p2 = self.SPP(p2)
        p3 = self.SPP(p3)
        
        return p1,p2,p3







class YOLOX(nn.Module):
    # Initialze the model
    # Inputs:
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
    def __init__(self, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, SPPDim, numCats):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.k = k
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        
        # The darknet backbone
        self.darknet = Darknet53(SPPDim)
        
        # Use 1x1 convolutions so that each value will have
        # the exact same dimensions
        self.conv11_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv11_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv11_3 = nn.Conv2d(256, 256, kernel_size=1)
        
        # The class convolution
        self.clsConv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, numCats, kernel_size=1)
        )
        
        # The regression convolution
        self.regConv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        
        # The regression 1x1
        self.reg11 = nn.Conv2d(256, 4, kernel_size=1)
        
        # The IoU 1x1
        self.iou11 = nn.Conv2d(256, 1, kernel_size=1)
        
        # Create the optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr=lr_init, momentum=momentum, weight_decay=weightDecay)
    
    
    # Get a prediction for the bounding box given some images
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    def forward(self, X):
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=device, requires_grad=False)
        
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
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=device, requires_grad=False)
        
        # Update the models `numEpochs` number of times
        for epoch in range(0, self.numEpochs):
            # Get a randomized set of indices
            idx = torch.randperm(X.shape[0])
            
            # Randomly split the data into batches
            X_batches = torch.split(X[idx], self.batchSize)
            y_temp = np.array(y, dtype=object)[idx]
            y_batches = [y_temp[self.batchSize*i + self.batchSize*(i+1)] for i in range(0, (X.shape[0]//self.batchSize))]
            y_batches = np.concatenate((y_batches, y_temp[(X.shape[0]//self.batchSize):]))
            
            # Iterate over all batches
            for i in range(0, len(X_batches)):
                # Get a prediction of the bounding boxes on the input image
                cls, reg, iou = self.forward(X)
                
                ### Class predictions
                print() # Should I argmax the classes to find which one
                # it predicted for each pixel in the img?
                
                # ALso should HxW go to the actual HxW of the original picture
                
                ### Regression predictions
                print()
                
                ### IoU predictions
                print()

        return 0