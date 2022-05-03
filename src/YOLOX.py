from Darknet53 import Darknet53
from LossFunctions import LossFunctions
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import math
import os
import json


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
    #   ImgDim - The height and width dimensions of the input image
    #   numCats - The number of categories to predict from
    #   FL_alpha - The focal loss alpha parameter
    #   FL_gamma - The focal loss gamma parameter
    #   reg_consts - The regression constraints (should be 4 values)
    #   reg_weight - Percent to weight the regression loss over the other loss
    def __init__(self, device, k, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_consts, reg_weight):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.k = k
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        self.device = device
        self.ImgDim = ImgDim
        assert len(reg_consts) == 4, "The regression constrainsts should contain 4 values"
        self.reg_consts = reg_consts
        self.numCats = numCats
        self.reg_weight = reg_weight
        
        # Trainable paramters for the exponential function which the
        # regression values are sent through
        self.exp_params = nn.ParameterList([nn.Parameter(torch.tensor(1, device=device, dtype=torch.float, requires_grad=True)) for i in range(0, 3)])
        
        # The stride to move each bounding box by for each different
        # level in the FPN (feature pyramid network)
        self.strides = [32, 16, 8]
        
        # The feature image shapes which are the three outputs
        # of the network
        self.FPNShapes = [ImgDim//self.strides[0], ImgDim//self.strides[1], ImgDim//self.strides[2]]
        
        # The position of each pixel for each level of the FPN
        # The FCOS formula is used: 
        #    [stride/2 + featx * stride, stride/2 + featy * stride]
        #    where featx and featy are (x, y) coordinates on the feature image
        #    where the feature image is one of the outputs of the FPN
        # These value will be used to directly map the regression values
        # back to the image
        self.FPNPos = [torch.tensor([[(self.strides[i]/2 + k * self.strides[i], self.strides[i]/2 + j * self.strides[i]) for k in range(0, self.FPNShapes[i])] for j in range(0, self.FPNShapes[i])], device=cpu, dtype=torch.long) for i in range(0, len(self.strides))]
        
        # The JSON data to save when saving the model
        self.JSON_Save = {
            "k": self.k,
            "ImgDim": self.ImgDim,
            "numCats": self.numCats,
            "strides": self.strides
        }
        
        # The darknet backbone and output head
        self.darknet = Darknet53(device, numCats+1)
        
        # The loss functions
        self.losses = LossFunctions(FL_gamma, FL_alpha)
        
        # Create the optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr_init*batchSize/64, momentum=momentum, weight_decay=weightDecay)
    
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=numEpochs)
    
    
    # Get a prediction for the bounding box given some images
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    # Outputs:
    #   Three arrays each with three elements. The first array is the class
    #   predictions, the second is the regression predictions, and the
    #   third is the objectiveness (IoU) predictions.
    def forward(self, X):
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Send the inputs through the Darknet backbone
        FPN1, FPN2, FPN3 = self.darknet(X)
        
        # Return the data as arrays
        return [FPN1[0], FPN2[0], FPN3[0]], [FPN1[1], FPN2[1], FPN3[1]], [FPN1[2], FPN2[2], FPN3[2]]
    
    
    
    # Setup the positive labels to determine whether
    # a boundary box should be used for each image
    # Input:
    #   GT - Ground truth information about a batch on images
    # Output:
    #   Labels for each level in the FPN where a positive label is
    #   1 and the negative label is 0. The output is of shape:
    #   (3, batchSize, FPN Size)
    def setupPos(self, GT):
        # The label information
        labels = []
        
        # Iterate over all FPN levels
        for lev in range(0, len(self.FPNShapes)):
            # Labels for this FPN level
            lev_lab = torch.zeros(len(GT), self.FPNShapes[lev]*self.FPNShapes[lev])
            
            # Get the FPN Position for this level
            pos = self.FPNPos[lev].reshape(self.FPNPos[lev].shape[0]*self.FPNPos[lev].shape[1], self.FPNPos[lev].shape[-1])
            
            # Iterate over all images
            for img in range(0, len(GT)):
                # Get the image information
                info = GT[img]['pix_cls']
                
                # Iterate over all coordinates which have a pixel
                # value which will predict a bounding box
                for p in range(0, pos.shape[0]):
                    # Sample a 3x3 area for an object. If an object
                    # is found, store a 1, else store a 0
                    if (True in (info[pos[p, 0], pos[p, 1]-1:pos[p, 1]+2] != 0)): # Row 1
                        lev_lab[img, p] = 1
                    elif (True in (info[pos[p, 0]+1, pos[p, 1]-1:pos[p, 1]+2] != 0)): # Row 2
                        lev_lab[img, p] = 1
                    elif (True in (info[pos[p, 0]-1, pos[p, 1]-1:pos[p, 1]+2] != 0)): # Row 3
                        lev_lab[img, p] = 1
                    # Do nothing if there is no class in a 3x3 area
                    
            # Store the labels
            labels.append(lev_lab)
        
        # Return the labels
        return labels
    
    
    
    # Get all ground truth targets for each bounding box
    # prediction
    # Inputs:
    #   GT - Ground truth information about a batch on images
    #   reg_labels_init - Regression label stating which regression
    #                     values are positive and negative
    # Outputs:
    #   targets - The ground truth bounding boxes for each
    #             prediction. The value is -1 if there is not bounding
    #             box to predict to save space. It has shape:
    #             (numFPNLevels, numImages, FPNsize_x, FPNSize_y, 4)
    def getTargets(self, GT, reg_labels_init):
        # The target information
        targets = []
        
        # Iterate over all FPN levels
        for level in range(0, 3):
            # Get the FPN level stride
            stride = self.strides[level]
            
            # Get the current FPN coordinates for this level
            coords = self.FPNPos[level]
            
            # array to store the GT bounding box label
            # for each bounding box prediction
            targs = torch.negative(torch.ones((len(GT), *coords.shape[:-1], 4), dtype=torch.int16, device=cpu, requires_grad=False))
            
            # Iterate over all images in the batch
            for img in range(0, len(GT)):
                
                # Get the image information
                img_info = GT[img]
                
                # Iterate over all FPN coordinates
                for c_x in range(0, len(coords)):
                    for c_y in range(0, len(coords[c_x])):
                        x = coords[c_x, c_y][0]
                        y = coords[c_x, c_y][1]
                        
                        # If there isn't an object in this pixel, store
                        # an array of -1s
                        if reg_labels_init[level][img, (c_x*int(math.sqrt(reg_labels_init[level].shape[1])))+c_y] == 0:
                            continue
                            
                        # Best bounding box and it's area.
                        # Note: we want to smallest bounding box
                        # this pixel is in
                        best_bbox = -1
                        best_bbox_area = torch.inf
                            
                        # Iterate over all bounding boxes for this label
                        for bbox_idx in range(len(img_info['bbox'])):
                            bbox = img_info['bbox'][bbox_idx]
                            
                            # If the pixel falls within a 3x3 area of the bounding box, 
                            # store it if the area is the smallest
                            if x+1 >= bbox[0] and x-1 <= bbox[0]+bbox[2] and \
                                y+1 >= bbox[1] and y-1 <= bbox[1]+bbox[3]:
                                # Get the area of the bounding box
                                area = (bbox[2]) * (bbox[3])
                                
                                # If the area is smaller than the current best,
                                # store this bounding box
                                if area < best_bbox_area:
                                    best_bbox_area = area
                                    best_bbox = bbox
                        
                        # If a boudning box was found, store it
                        if best_bbox != -1:
                            targs[img, c_x, c_y] = torch.tensor(best_bbox, device=cpu, requires_grad=False)
            
            # Add this FPN targets to the total targets
            targets.append(targs.reshape(targs.shape[0], targs.shape[1]*targs.shape[2], targs.shape[3]))
        
        return targets

    
    
    
    # Decode the regression outputs so that
    # they are moved to the correct location on the input image
    # Inputs:
    #   regs - The regression value to normalize of shape:
    #     (batchSize, FPNLevel Size, 4)
    #     - The 4 values are x, y, width, and height
    #   p - The FPN level the predictions came from
    def regDecode(self, regs, p):
        # The new predictions
        new_preds = []
        
        # Iterate over all images
        for img in range(0, regs.shape[0]):
            # Predictions for the image
            img_preds = []
            
            # Iterate over all predictions
            for pred in range(0, regs.shape[1]):
                # Array of newly decoded values
                decoded = []
                
                # Move the x and y values to their proper location
                # on the original image
                decoded.append(regs[img, pred, 0:2] + self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[2])[pred])
                
                # First exponentiate the w and h so that they are not negative
                temp = torch.exp(self.exp_params[p]*regs[img, pred, 2:])
                
                # Move the w and h to their proper location
                decoded.append(temp * self.strides[p])
                
                # Save this prediction
                img_preds.append(torch.cat(decoded))
            
            # Save the predictions for this image
            new_preds.append(torch.stack(img_preds))
        
        # Return the new regression values
        return torch.stack(new_preds)
    
    
    # Train the network on training data
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    #   y - The labels for each input (correct bounding boxes to place on image)
    #   saveParams - Model saving paramters in list format with the following items:
    #       - [saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite]
    def train(self, X, y, saveParams):
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=cpu, requires_grad=False)
        
        # For each image, setup the bounding box labels so
        # we know which ones to count as positives
        reg_labels_init = self.setupPos(y)
        
        # Get the regression targets for each bounding
        # box in the input image
        reg_targets = self.getTargets(y, reg_labels_init)
        
        # Unpack the save parameters
        saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite = saveParams
        
        # The best loss so far
        bestLoss = torch.inf
        
        # Update the models `numEpochs` number of times
        for epoch in range(1, self.numEpochs+1):
            # Get a randomized set of indices
            idx = torch.randperm(X.shape[0], device=cpu)
            
            # Randomly split the data into batches
            X_batches = torch.split(X[idx], self.batchSize)
            y_tmp = np.array(y, dtype=object)[idx]
            y_batches = np.array([y_tmp[i*self.batchSize:(i+1)*self.batchSize] for i in range(y_tmp.shape[0]//self.batchSize)] + [y_tmp[(y_tmp.shape[0]//self.batchSize)*self.batchSize:]], dtype=object)
            reg_labels_init_batches = [torch.split(reg_labels_init[i][idx], self.batchSize) for i in range(0, len(reg_labels_init))]
            reg_targets_batches = [torch.split(reg_targets[i][idx], self.batchSize) for i in range(0, len(reg_targets))]
            
            # The loss over all batches
            batchLoss = 0
            
            # Iterate over all batches
            for batch in range(0, len(X_batches)):
                # Load the batch data
                X_b = X_batches[batch].to(self.device)
                y_b = y_batches[batch]
                reg_labels_b = [reg_labels_init_batches[i][batch] for i in range(0, len(reg_labels_init))]
                reg_targets_b = [reg_targets_batches[i][batch] for i in range(0, len(reg_targets))]
                
                # Get a prediction of the bounding boxes on the input image
                cls, reg, iou = self.forward(X_b)
                
                # Cumulate the loss across the three predictions
                totalLoss = 0
                
                # Iterate over the three sets of predictions
                # which is each FPN level
                for p in range(0, 3):
                    
                    ### Loss Setup ###
                    
                    # Get the current set of predictions
                    cls_p = cls[p].permute(0, 2, 3, 1)
                    reg_p = reg[p].permute(0, 2, 3, 1)
                    obj_p = iou[p].permute(0, 2, 3, 1)
                    
                    # Flatten all predictions
                    cls_p = cls_p.reshape(cls_p.shape[0], cls_p.shape[1]*cls_p.shape[2], cls_p.shape[3])
                    reg_p = reg_p.reshape(reg_p.shape[0], reg_p.shape[1]*reg_p.shape[2], reg_p.shape[3])
                    obj_p = obj_p.reshape(obj_p.shape[0], obj_p.shape[1]*obj_p.shape[2], obj_p.shape[3])
                    
                    # Decode the regression outputs:
                    # The output of the regression is:
                    #   [x, y, w, h,]
                    # We want to exponentiate the w and h so it cannot
                    # be negative and we want to move all values
                    # to their correct location based on the stride
                    # of the image
                    reg_p = self.regDecode(reg_p, p)
                    
                    # Get the positive filtered labels for
                    # this FPN level. These will change based
                    # on how good each prediction is
                    reg_labels = reg_labels_b[p]
                    
                    # Copy the regression labels to make objectiveness
                    # predictions
                    obj_labels = torch.clone(reg_labels)
                    
                    # Get the regression targets for
                    # this FPN level
                    reg_targs = reg_targets_b[p]
                    
                    # Denormalize/move the predicted bounding boxes
                    # so they are in the correct location
                    # pos = self.FPNPos[p]
                    # for x_mov in range(0, reg_p.shape[1]):
                    #     for y_mov in range(0, reg_p.shape[2]):
                    #         reg_p[:, x_mov, y_mov, 0] += pos[x_mov, y_mov, 0]
                    #         reg_p[:, x_mov, y_mov, 1] += pos[x_mov, y_mov, 1]
                    
                    
                    ### Positive Filtering
                    
                    #reg_labels = self.filterPos(reg_p, p, y_b)
                    
                    # Create a matrix of positive/negative targets. A positive target
                    # is one which is within a certain threshold. So, the width and
                    # height should be within a predefined threshold.
                    #reg_labels = torch.where(torch.logical_and(torch.max(reg_p[:,:,2:], dim=-1)[0] < reg_const_high, torch.min(reg_p[:,:,2:], dim=-1)[0] > reg_const_low), 1, 0)
                    
                    
                    
                    
                    
                    
                    ### Regression Loss 
                    
                    # The total GIoU loss
                    GIoU_loss = 0
                    
                    # Get the GIoU between each target bounding box
                    # and each predicted bounding box
                    
                    # Iterate over all batch elements
                    for b_num in range(0, obj_p.shape[0]):
                        # The predicted bounding boxes
                        bbox = reg_p[b_num]
                        
                        # Move the prediced bounding box to the location
                        # on the image
                        #bbox_conv = []
                        #for i in range(bbox.shape[0]):
                        #    box = [0, 0, 0, 0]
                        #    box[0] = bbox[i][0]+self.FPNPos[p][math.floor(i/self.FPNPos[p].shape[0]), i%self.FPNPos[p].shape[0], 0]
                        #    box[1] = bbox[i][1]+self.FPNPos[p][math.floor(i/self.FPNPos[p].shape[0]), i%self.FPNPos[p].shape[0], 1]
                        #    box[2] = bbox[i][2]
                        #    box[3] = bbox[i][3]
                        #    bbox_conv.append(torch.stack(box))
                        #bbox_conv = torch.stack(bbox_conv)
                        
                        # # Indices for each predicted bounding box
                        # # ground truth value
                        GTs = reg_targs[b_num]
                        # GTs = torch.clone(reg_targs[b_num])
                        
                        # # Get the GT bounding boxes as the corresponding
                        # # train value which is the boudning box coord
                        # # minus the pixel location of the prediction
                        # # (according to FCOS: x - x_0^i)
                        # #GT_bbox = torch.tensor(GTs[b_num]['bbox'], dtype=float, requires_grad=False, device=cpu)
                        # for i in range(0, int(math.sqrt(GTs.shape[0]))):
                        #     for j in range(0, int(math.sqrt(GTs.shape[0]))):
                        #         coord = self.FPNPos[p][i, j]
                        #         loc = (i*int(math.sqrt(GTs.shape[0])))+j
                                
                        #         # Update the value if the GT box is not all -1s
                        #         if GTs[loc][0] != -1:
                        #             GTs[loc][0] = coord[0] - GTs[loc][0]
                        #             GTs[loc][1] = coord[1] - GTs[loc][1]
                        # #GTs = [GTs[i]-self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[2]) for i in range(bbox.shape[0])]
                        # #GT_bbox = torch.stack(GT_bbox)
                        
                        # If there are no positive targets for this image,
                        # skip this iteration
                        if bbox[reg_labels[b_num] != 0].shape[0] == 0:
                            continue
                        
                        # Get the GIoU Loss and Value for the positive targets
                        GIoULoss, _ = self.losses.GIoU(bbox[reg_labels[b_num] != 0].to(cpu), GTs[reg_labels[b_num] != 0])
                        
                        # Sum the loss across all images and save it
                        GIoU_loss += GIoULoss.sum()
                    
                    
                    
                    ### Class Loss
                    
                    # Get the ground truth value for each prediction as a one-hot vector
                    GT = torch.zeros(list(cls_p.shape[:-1])+[self.numCats+1])
                    
                    # Iterate over all images
                    temp = self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[2])
                    for b_num in range(0, cls_p.shape[0]): # Iterate over all batch elements
                        # Iterate over all FPN positions
                        for pos in range(0, self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1]):
                            # Current coords mapping back to the original image
                            coord = temp[pos]
                            
                            # Is there a class other than the background class (0) in
                            # a 3x3 area?
                            first = y_b[b_num]['pix_cls'][coord[0], coord[1]-1:coord[1]+2]
                            second = y_b[b_num]['pix_cls'][coord[0]+1, coord[1]-1:coord[1]+2]
                            third = y_b[b_num]['pix_cls'][coord[0]-1, coord[1]-1:coord[1]+2]
                            
                            # Get the possible classes
                            pos_cls = torch.cat((first, second, third))
                            
                            # Pick the highest class and store it as a 1 in a one-hot vector
                            GT[b_num, pos][torch.max(pos_cls)] = 1
                    
                    # Send the data through the Focal Loss function
                    #cls_Loss = self.losses.FocalLoss(cls_p.to(cpu), GT)
                    
                    # Send the positive data through the BCE loss function
                    cls_Loss = self.losses.BinaryCrossEntropy(cls_p[reg_labels != 0].to(cpu), GT[reg_labels != 0])
                    #cls_Loss = nn.BCEWithLogitsLoss(reduction='sum')(cls_p.to(cpu)[reg_labels != 0], GT[reg_labels != 0])
                    
                    
                    
                    
                    ### Objectiveness Loss
                    
                    
                    # Total objectiveness loss for the bounding boxes
                    obj_Loss = 0
                    
                    # Iterate over all elements in the batch
                    for b_num in range(0, len(y_b)):
                        ## Get a tensor of all the ground truth objective values.
                        ## Basically, this value is the GIoU value for each
                        ## predicted bounding box
                        
                        # Get the current objctiveness predictions
                        obj = torch.squeeze(obj_p[b_num])
                        
                        # Don't calculate the gradient when finding the GT values
                        with torch.no_grad():
                            # Get all ground truth bounding boxes in this image
                            GT_bbox = torch.tensor(y_b[b_num]["bbox"], device=cpu, requires_grad=False)
                            
                            # The best GIoU values for each predicted bounding box
                            best_GIoU = torch.negative(torch.ones(obj.shape, requires_grad=False, device=cpu, dtype=torch.float))
                            
                            # Iterate over all GT boxes
                            for box in GT_bbox:
                                # Broadcast the GT box
                                box = torch.broadcast_to(box, reg_p[b_num].shape).to(cpu).clone()
                                
                                # For those boxes with a GT box, use those boxes instead
                                # of the closest box
                                box[reg_targs[b_num, :, 0] != -1] = reg_targs[b_num].long()[torch.where(reg_targs[b_num, :, 0] != -1)[0]]
                                
                                # Get the predicted bounding boxes. Note, these have
                                # already been projected to the original image
                                pred_bbox = reg_p[b_num].to(cpu)
                                
                                # Get the GIoU between the predicted boudning boxes
                                # and the current bounding box in iteration
                                _, all_GIoU = self.losses.GIoU(pred_bbox, box)
                                
                                # Get the max GIoU loss for each bounding box
                                # between the new GIoU values and the current
                                # saved ones. Save the max values
                                best_GIoU = torch.maximum(best_GIoU, 1/(1+torch.exp(-5*all_GIoU)))
                        
                        # Get the loss between the batch elements
                        # Note: We don't just want the positively
                        # labelled ones since we want the model to learn
                        # both bad and good predictions
                        obj_Loss += self.losses.BinaryCrossEntropy(obj.to(cpu), best_GIoU)
                        #iouLoss += self.losses.BinaryCrossEntropy(obj[reg_labels[b_num] == 1].to(cpu), GTobj[reg_labels[b_num] == 1])
                        
                        
                        
                        
                        # # Get the current IoU (objctiveness) predictions
                        # # for all positive labels
                        # obj_sub = torch.squeeze(obj_p[b_num])[reg_labels[b_num] != 0].to(cpu)
                        
                        # # If there are no predictions, skip this iteration
                        # if obj_sub.shape[0] == 0:
                        #     continue
                        
                        # # Get the objectiveness loss for this image
                        # obj_Loss += self.losses.BinaryCrossEntropy(obj_sub, saved_GIoU[b_num])
                    
                    
                    
                    
                    
                    
                    
                    ### Final Loss ###
                    
                    # Get the final loss for this prediction
                    N_pos = torch.count_nonzero(reg_labels)
                    finalLoss = (1/N_pos)*cls_Loss + self.reg_weight*((1/N_pos)*GIoU_loss) + (1/N_pos)*obj_Loss
                    #finalLoss = (1/N_pos)*obj_Loss
                    totalLoss += finalLoss
                
                
                ### Updating the model
                #if epoch%20 == 0:
                #    plt.imshow(torch.argmax(cls_p[0], dim=-1).reshape(int(cls_p[0].shape[0]**0.5), int(cls_p[0].shape[0]**0.5)).cpu().detach().numpy(), interpolation='nearest')
                #    plt.show()
                #if epoch%5 == 0:
                #    plt.imshow(iou_p[0].cpu().detach().numpy(), interpolation='nearest')
                #    plt.show()
                
                # Backpropogate the loss
                totalLoss.backward()
                
                # Clip the gradients so that the model doesn't
                # go too crazy when updating its parameters
                #torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                
                # Step the optimizer
                self.optimizer.step()
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Update the batch loss
                batchLoss += totalLoss.cpu().detach().numpy().item()
            
            # Convert the printed boundary boxes to their proper form
            print_boxes = reg_p[2]
            #for bbox in range(print_boxes.shape[0]):
            #    print_boxes[bbox, 0] = print_boxes[bbox, 0]+self.FPNPos[p][math.floor(bbox/self.FPNPos[p].shape[0]), bbox%self.FPNPos[p].shape[0], 0]
            #    print_boxes[bbox, 1] = print_boxes[bbox, 1]+self.FPNPos[p][math.floor(bbox/self.FPNPos[p].shape[0]), bbox%self.FPNPos[p].shape[0], 1]
            
            print(f"Step #{epoch}      Total Batch Loss: {batchLoss}")
            print("Reg:")
            print(f"Prediction: {print_boxes[reg_labels[2] == 1][:2].cpu().detach().numpy()}")
            print(f"Ground Truth: {reg_targs[2][reg_labels[2] == 1][:2].cpu().detach().numpy()}")
            
            cls_GT = []
            for i in self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[-1]):
                first = y_b[2]['pix_cls'][i[0], i[1]-1:i[1]+2]
                second = y_b[2]['pix_cls'][i[0]-1, i[1]-1:i[1]+2]
                third = y_b[2]['pix_cls'][i[0]+1, i[1]-1:i[1]+2]
                best = torch.cat((first, second, third)).max()
                cls_GT.append(best)
            cls_GT = torch.stack(cls_GT)
            
            print()
            print("Cls:")
            print(f"Prediction: {torch.argmax(cls_p[2][reg_labels[2] == 1][:2], dim=1).cpu().detach().numpy()}")
            print(f"Ground Truth: {cls_GT[reg_labels[2] == 1][:2].cpu().detach().numpy()}")
            print()
            print("Obj:")
            print(f"Prediction: {obj_p[2][reg_labels[2] == 1][:2].cpu().detach().numpy()}")
            print("\n")
            
            # Step the learning rate scheduler after the warmup steps
            if epoch > self.warmupEpochs:
                self.scheduler.step()
            
            # Save the model if the model is in the proper state
            if saveSteps != 0:
                if epoch % saveSteps == 0:
                    if saveOnBest == True:
                        if batchLoss < bestLoss:
                            bestLoss = batchLoss
                            self.saveModel(saveDir, paramSaveName, saveName, overwrite, epoch)
                
                    else:
                        self.saveModel(saveDir, paramSaveName, saveName, overwrite, epoch)
        
        return 0
        
    
    
    
    
    # Save the model
    # Inputs:
    #   saveDir - The directory to save models to
    #   paramSaveName - The file to save the model paramters to
    #   saveName - File to save the model to
    #   overwrite - True to overwrite the file when saving.
    #               False to make a new file when saving
    #   epoch (optional) - The current epoch the model is on when training
    def saveModel(self, saveDir, paramSaveName, saveName, overwrite, epoch=0):        
        # Ensure the directory exists 
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        
        # If overwrite is False, create the new filename using the
        # current epoch by appending the epoch to the file name
        if not overwrite:
            modelSaveName = f"{saveName} - {epoch}"
            paramSaveName = f"{paramSaveName} - {epoch}"
            
        # Add .pkl to the end of the model save name
        modelSaveName = f"{modelSaveName}.pkl"
        
        # Add .json to the end of the paramater save name
        paramSaveName = f"{paramSaveName}.json"
        
        # Save the model
        torch.save(self.state_dict(), os.path.join(saveDir, modelSaveName))
        
        # Save the paramters
        with open(os.path.join(saveDir, paramSaveName), "w", encoding='utf-8') as f:
            json.dump(self.JSON_Save, f, ensure_ascii=False)
    
    
    
    
    # Load the model from a .pkl file
    # Input:
    #   loadDir - The directory to load the model from
    #   paramLoadName - The name of the file to load the model paramters from
    #   loadName - The name of the file to load the model from
    def loadModel(self, loadDir, paramLoadName, loadName):
        # Create the full file name
        modelFileName = os.path.join(loadDir, loadName)
        paramFileName = os.path.join(loadDir, paramLoadName)
        
        # Ensure the directory exists
        assert os.path.isdir(loadDir), f"Load directory {loadDir} does not exist."
        
        # Ensure the model file exists
        assert os.path.isfile(modelFileName), f"Load file {modelFileName} does not exist."
        
        # Ensure the parameter file exists
        assert os.path.isfile(paramFileName), f"Load file {paramFileName} does not exist."
        
        # Load in the model file if it exists
        self.load_state_dict(torch.load(modelFileName))
        
        # Load in the parameters
        with open(paramFileName, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        # Save the loaded data to the model
        self.k = data['k']
        self.ImgDim = data['ImgDim']
        self.numCats = data['numCats']
        self.strides = data['strides']
        self.JSON_Save = data
        self.FPNShapes = [self.ImgDim//self.strides[0], self.ImgDim//self.strides[1], self.ImgDim//self.strides[2]]
        self.FPNPos = [torch.tensor([[(self.strides[i]/2 + k * self.strides[i], self.strides[i]/2 + j * self.strides[i]) for k in range(0, self.FPNShapes[i])] for j in range(0, self.FPNShapes[i])], device=cpu, dtype=torch.long) for i in range(0, len(self.strides))]
    
    
    
    
    
    
    
    
    # Get predictions from the network on some images
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on) 
    def predict(self, X):
        #https://medium.com/swlh/fcos-walkthrough-the-fully-convolutional-approach-to-object-detection-777f614268c
        # Look in inference mode


        # Note: Send cls and obj output through sigmoid, but not the reg outputs


        return 1