import torch
from torch import nn


class conv(nn.Module):
    # A single convolution layer used by Darknet53
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):
        super(conv, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
        )
    
    def forward(self, X):
        return self.block(X)



class residual(nn.Module):
    # A residual block containing a 1x1 and a 3x3 convolution
    def __init__(self, in_chan):
        super(residual, self).__init__()
        
        self.block = nn.Sequential(
            conv(in_chan, in_chan//2, kernel_size=1, padding=0),
            conv(in_chan//2, in_chan, kernel_size=3)
        )
    
    def forward(self, X):
        res = X
        
        X_new = self.block(X)
        
        return X_new + res




class Darknet53(nn.Module):
    # Initialize the model
    # Inputs:
    #   device - Device to put the network on
    #   numClasses - Number of classes to predict
    #   inChanSize - The initial number of input channels (3 for RGB)
    def __init__(self, device, numClasses, inChanSize=3):
        super(Darknet53, self).__init__()
        
        ### Darknet blocks
        self.block1 = nn.Sequential(
            conv(inChanSize, 32, kernel_size=3),
            conv(32, 64, kernel_size=3, stride=2),
        ).to(device)
        self.res1 = residual(64).to(device)
        self.block2 = nn.Sequential(
            conv(64, 128, kernel_size=3, stride=2)
        ).to(device)
        self.res2 = nn.Sequential(
            residual(128),
            residual(128),
        ).to(device)
        self.block3 = nn.Sequential(
            conv(128, 256, stride=2)
        ).to(device)
        self.res3 = nn.Sequential(
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
            residual(256),
        ).to(device)
        self.block4 = nn.Sequential(
            conv(256, 512, stride=2)
        ).to(device)
        self.res4 = nn.Sequential(
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
            residual(512),
        ).to(device)
        self.block5 = nn.Sequential(
            conv(512, 1024, stride=2)
        ).to(device)
        self.res5 = nn.Sequential(
            residual(1024),
            residual(1024),
            residual(1024),
            residual(1024),
        ).to(device)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.connected = nn.Linear(1024, 1000, device=device)
        
        ### Output/prediction blocks
        
        # 1x1 block conversions
        
        # highest FPN level head (1024 channels)
        self.head1_1x1 = nn.Sequential(
            conv(1024, 256, kernel_size=1, padding=0),
        ).to(device)
        self.head1_cls = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
            nn.Conv2d(256, numClasses, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head1_reg_base = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
        ).to(device)
        self.head1_reg = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head1_IoU = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
        ).to(device)
        
        
        
        # Middle FPN level head (512 channels)
        self.head2_1x1 = nn.Sequential(
            conv(512, 256, kernel_size=1, padding=0),
        ).to(device)
        self.head2_cls = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
            nn.Conv2d(256, numClasses, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head2_reg_base = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
        ).to(device)
        self.head2_reg = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head2_IoU = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
        ).to(device)
        
        
        # Lowest FPN level head (256 channels)
        self.head3_1x1 = nn.Sequential(
            conv(256, 256, kernel_size=1, padding=0),
        ).to(device)
        self.head3_cls = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
            nn.Conv2d(256, numClasses, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head3_reg_base = nn.Sequential(
            conv(256, 256),
            conv(256, 256),
        ).to(device)
        self.head3_reg = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, padding=0, bias=False)
        ).to(device)
        self.head3_IoU = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False)
        ).to(device)
        
    
    # Get a prediction from the model
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    # Outputs:
    #   p1, p2, p3 - Predictions from different levels in the model where
    #                each prediction has 3 elements:
    #                [class, regression, IoU]
    def forward(self, X):
        ### Send the inputs through the darknet backbone
        X = self.block1(X)
        X = self.res1(X)
        X = self.block2(X)
        X = self.res2(X)
        X = self.block3(X)
        X = self.res3(X)
            
        # Save the first checkpoint (lowest prediction)
        p3 = torch.clone(X)
        
        X = self.block4(X)
        X = self.res4(X)
        
        # Save the second checkpoint (middle prediction)
        p2 = torch.clone(X)
        
        X = self.block5(X)
        X = self.res5(X)
        
        # Save the final checkpoint (highest prediction)
        p1 = torch.clone(X)
        
        X = torch.squeeze(self.globalPool(X))
        X = self.connected(X)
    
    
    
        ### Now send the three checkpoints through
        ### the YOLO head to get the outputs
        
        
        # Get the highest FPN level predictions (1024 channels)
        p1 = self.head1_1x1(p1)
        p1_cls = self.head1_cls(p1)
        p1 = self.head1_reg_base(p1)
        p1_reg = self.head1_reg(p1)
        p1_IoU = self.head1_IoU(p1)
        p1 = [p1_cls, p1_reg, p1_IoU]
        
        
        # Get the highest FPN level predictions (512 channels)
        p2 = self.head2_1x1(p2)
        p2_cls = self.head2_cls(p2)
        p2 = self.head2_reg_base(p2)
        p2_reg = self.head2_reg(p2)
        p2_IoU = self.head2_IoU(p2)
        p2 = [p2_cls, p2_reg, p2_IoU]
        
        
        # Get the highest FPN level predictions (1024 channels)
        p3 = self.head3_1x1(p3)
        p3_cls = self.head3_cls(p3)
        p3 = self.head3_reg_base(p3)
        p3_reg = self.head3_reg(p3)
        p3_IoU = self.head3_IoU(p3)
        p3 = [p3_cls, p3_reg, p3_IoU]
        
        return p1,p2,p3