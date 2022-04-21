import torch
from torch import nn




class Darknet53(nn.Module):
    # Initialize the mode
    # Inputs:
    #   device - Device to put the network on
    #   SPPDim - The dimension to convert each image to (output image will be (C x SPPDim x SPPDim))
    #   inChanSize - The number of input channels (3 for RGB)
    def __init__(self, device, SPPDim, inChanSize=3):
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