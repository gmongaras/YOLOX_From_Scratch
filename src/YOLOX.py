from ast import Delete
import torch
from torch import nn


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')






class Darknet53(nn.Module):
    # Initialize the mode
    # Inputs:
    #   inSize - The number of input channels (3 for RGB)
    def __init__(self, inSize=3):
        super(Darknet53, self).__init__()
        
        # Create the model blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(inSize, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
        ).to(device)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        ).to(device)
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        ).to(device)
        self.block7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
        ).to(device)
        self.block9 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
        ).to(device)
        self.block10 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        ).to(device)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.connected = nn.Linear(1024, 1000)
        self.soft = nn.Softmax(0)
        
    
    # Get a prediction from the model
    def forward(self, X):
        v = self.block1(X)
        
        res = torch.clone(v)
        v = self.block2(v)+res
        v = self.block3(v)
        
        res = torch.clone(v)
        v = self.block4(v)+res
        res = torch.clone(v)
        v = self.block4(v)+res
        
        v = self.block5(v)
        
        for i in range(0, 8):
            res = torch.clone(v)
            v = self.block6(v)+res
        
        v = self.block7(v)
        
        for i in range(0, 8):
            res = torch.clone(v)
            v = self.block8(v)+res
        
        v = self.block9(v)
        
        for i in range(0, 4):
            res = torch.clone(v)
            v = self.block10(v)+res
        del res
        
        v = torch.squeeze(self.globalPool(v))
        
        v = self.connected(v)
        
        v = self.soft(v)
        
        return v







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
    def __init__(self, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.k = k
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        
        # Create the optimizer
        self.optim = torch.optim.SGD(self.parameters, lr=lr_init, momentum=momentum, weight_decay=weightDecay)
    
    
    
    def forward(self):
        return 1