from Darknet53 import Darknet53
from LossFunctions import LossFunctions

import torch
from torch import nn
import numpy as np
import math
import os
import json
from random import randint
import skimage.io as io
from copy import deepcopy

from nonmaxSupression import soft_nonmaxSupression
from SimOTA import SimOTA

from dataAugmentation import Mosaic
from dataAugmentation import mixup
from PIL import Image


cpu = torch.device('cpu')
gpu = torch.device('cuda:0')




class YOLOX(nn.Module):
    # Initialze the model
    # Inputs:
    #   device - Device to put the network on
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
    #   reg_weight - Percent to weight the regression loss over the other loss
    #   category_Ids - Dictionary mapping categories to their ids
    #   removal_threshold - The threshold of predictions to remove if the
    #                       confidence in that prediction is below this value
    #   score_thresh - The score threshold to remove boxes. If the score is
    #                  less than this value, remove it
    #   IoU_thresh - The IoU threshold to update scores. If the IoU is
    #                greater than this value, update it's score
    #   nonmax_threshold - The threshold of predictions to remove if the
    #                       IoU is under this threshold
    #   SimOTA_params - Paramters used for the SimOta calculation:
    #   1. q - The number of GIoU values to pick when calculating the k values
    #   2. r - The radius used to calculate the center prior
    #   3. extraCost - The extra cost used in the center prior computation
    #   4. SimOta_lambda = Balancing factor for the foreground loss
    #   - This parameter is only required for training
    def __init__(self, device, numEpochs, batchSize, warmupEpochs, lr_init, weightDecay, momentum, ImgDim, numCats, FL_alpha, FL_gamma, reg_weight, category_Ids, removal_threshold, score_thresh, IoU_thresh, SimOTA_params=None):
        super(YOLOX, self).__init__()
        
        # Save the model paramters
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.warmupEpochs = warmupEpochs
        self.lr_init = lr_init
        self.device_str = device.lower()
        self.ImgDim = ImgDim
        self.numCats = numCats
        self.reg_weight = reg_weight
        self.category_Ids = category_Ids
        self.removal_threshold = removal_threshold
        self.score_thresh = score_thresh
        self.IoU_thresh = IoU_thresh
        self.SimOTA_params = SimOTA_params

        # Get the device to put tensors on
        if self.device_str == "fullgpu" or self.device_str == "gpu":
            self.device = gpu
        else:
            self.device = cpu
        
        # Trainable paramters for the exponential function which the
        # regression values are sent through
        self.exp_params = nn.ParameterList([nn.Parameter(torch.tensor(1, device=self.device, dtype=torch.float, requires_grad=True)) for i in range(0, 3)])
        
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
        self.FPNPos = [torch.tensor([[(self.strides[i]/2 + k * self.strides[i], self.strides[i]/2 + j * self.strides[i]) for k in range(0, self.FPNShapes[i])] for j in range(0, self.FPNShapes[i])], device=self.device, dtype=torch.long) for i in range(0, len(self.strides))]
        
        # The JSON data to save when saving the model
        self.JSON_Save = {
            "ImgDim": self.ImgDim,
            "numCats": self.numCats,
            "strides": self.strides,
            "category_Ids": self.category_Ids
        }
        
        # The darknet backbone and output head
        self.darknet = Darknet53(self.device, numCats+1)
        
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
        
        # Make sure the input datatype is float32
        if X.dtype != torch.float32:
            X = X.float()
        
        # Send the inputs through the Darknet backbone
        FPN1, FPN2, FPN3 = self.darknet(X)
        
        # Return the data as arrays
        return [FPN1[0], FPN2[0], FPN3[0]], [FPN1[1], FPN2[1], FPN3[1]], [FPN1[2], FPN2[2], FPN3[2]]

    
    
    
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
        
        # Flatten the FPN Positions
        FPNPos_flat = self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[2])
        
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
                decoded.append(regs[img, pred, 0:2] + FPNPos_flat[pred])
                
                # Begin moving the regression predictions by moving
                # each prediction by the exponent constant
                moved_preds = self.exp_params[p]*regs[img, pred, 2:]
                
                # Don't allow the exponentiated values above half
                # the image size to avoid the values being
                # blown up and causing nans
                moved_preds = torch.clamp(moved_preds, 0, np.log(self.ImgDim//2))
                
                # Now exponentiate the w and h so that they are not negative
                moved_preds = torch.exp(moved_preds)
                
                # Move the w and h to their proper location
                decoded.append(moved_preds * self.strides[p])
                
                # Save this prediction
                img_preds.append(torch.cat(decoded))
            
            # Save the predictions for this image
            new_preds.append(torch.stack(img_preds))
        
        # Return the new regression values
        return torch.stack(new_preds).to(self.device)
    
    
    # Train the network on training data
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on)
    #   y - The labels for each input (correct bounding boxes to place on image)
    #   dataAug_params - A list containing four elements:
    #   1. directory storing images to load in
    #   2. img_data - Loaded data from the Coco dataset on images
    #   3. ann_data - Loaded data from the Coco dataset on annotations
    #   4. category_Ids - A dictionary mapping the id of a Coco object
    #                     to the name of that object
    #   augment_per - Percent of extra data to generate every epoch
    #   saveParams - Model saving paramters in list format with the following items:
    #       - [saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite]
    def train_model(self, X, y, dataAug_params, augment_per, saveParams):
        # Put the model in training mode
        self.train()
        
        # Make sure the input data are tensors
        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Unpack the save parameters
        saveDir, paramSaveName, saveName, saveSteps, saveOnBest, overwrite = saveParams
        
        # Unpack the augmentation paramters
        img_dir, img_data, ann_data, category_Ids = dataAug_params
        
        # The best loss so far
        bestLoss = torch.inf
        
        # The original shape of the input
        origShape = X.shape[0]
        
        # Update the models `numEpochs` number of times
        for epoch in range(1, self.numEpochs+1):
            
            # If X is not the original shape, then remove the
            # previously generated agumented data
            if X.shape[0] != origShape:
                tmp = X[origShape:]
                X = X[:origShape]
                del tmp
                
                tmp = y[origShape:]
                y = y[:origShape]
                del tmp
            
            # Get the number of data to generate and augment
            numToAugment = int(X.shape[0]*augment_per)
            
            # The new images and annotations from
            # all augmentations
            new_imgs = []
            new_anns = []
            
            # Half the data will be generated by Mosaic
            for aug in range(0, numToAugment//2):
                # The images to augment with Moasic
                aug_imgs = []
                aug_anns = []
                
                # Get 4 random images that have been loaded in
                for im in range(0, 4):
                    # Get a random image from the list loaded images
                    rand_idx = math.floor(randint(0, X.shape[0]-1))
                    
                    # Load in the image and it's annotations
                    aug_img = io.imread(img_dir + img_data[rand_idx]["file_name"])
                    aug_ann = {"bbox":deepcopy([i['bbox'] for i in ann_data[img_data[rand_idx]["id"]]]),
                               "cls":[list(category_Ids.values()).index(i["category_id"])+1 for i in ann_data[img_data[rand_idx]["id"]]]}
                    
                    # Convert the image to RGB
                    aug_img = np.array(Image.fromarray(aug_img).convert("RGB"))
                    
                    # Save the image and annotations
                    aug_imgs.append(aug_img)
                    aug_anns.append(aug_ann)
                
                # Using the 4 images, generate a new image and it's
                # annotations until there is atleast 1 annotation in the
                # resulting image
                new_ann = []
                while len(new_ann) == 0:
                    new_img, new_ann = Mosaic(aug_imgs, aug_anns, (X.shape[2], X.shape[3]))
                
                new_imgs.append(new_img)
                new_anns.append(new_ann)
            
            # The other half is generated by MixUp
            for aug in range(0, numToAugment-(numToAugment//2)):
                # The images to augment with mixUp
                aug_imgs = []
                aug_anns = []
                
                # Get 2 random images that have been loaded in
                for im in range(0, 2):
                    # Get a random image from the list loaded images
                    rand_idx = math.floor(randint(0, X.shape[0]-1))
                    
                    # Load in the image and it's annotations
                    aug_img = io.imread(img_dir + img_data[rand_idx]["file_name"])
                    aug_ann = {"bbox":deepcopy([i['bbox'] for i in ann_data[img_data[rand_idx]["id"]]]),
                               "cls":[list(category_Ids.values()).index(i["category_id"])+1 for i in ann_data[img_data[rand_idx]["id"]]]}
                    
                    # Convert the image to RGB
                    aug_img = np.array(Image.fromarray(aug_img).convert("RGB"))
                    
                    # Save the image and annotations
                    aug_imgs.append(aug_img)
                    aug_anns.append(aug_ann)
                
                # Using the 2 images, generate a new image and it's
                # annotations
                new_img, new_ann = mixup(*aug_imgs, *aug_anns, (X.shape[2], X.shape[3]))
                
                new_imgs.append(new_img)
                new_anns.append(new_ann)

            
            # Store the images and annotations
            stk = torch.stack(new_imgs).to(self.device)
            X = torch.cat((X, stk)).to(self.device)
            y += new_anns
            
            
            # Get a randomized set of indices
            idx = torch.randperm(X.shape[0], device=cpu)
            
            # Randomly split the data into batches
            X_batches = torch.split(X[idx], self.batchSize)
            y_tmp = np.array(y, dtype=object)[idx]
            y_batches = np.array([y_tmp[i*self.batchSize:(i+1)*self.batchSize] for i in range(y_tmp.shape[0]//self.batchSize)] + [y_tmp[(y_tmp.shape[0]//self.batchSize)*self.batchSize:]], dtype=object)
            
            # The loss over all batches
            batchLoss = 0
            
            # Iterate over all batches
            for batch in range(0, len(X_batches)):
                # Load the batch data
                X_b = X_batches[batch]
                y_b = y_batches[batch]

                # If partial GPU support is used, put the input data
                # on the GPU before going through the model
                if self.device == "partgpu":
                    X_b = X_b.to(gpu)
                
                # Get a prediction of the bounding boxes on the input image
                cls, reg, iou = self.forward(X_b)

                # If partial GPU support is used, take the data off the GPU
                # and put it on the CPU
                if self.device == "partgpu":
                    cls = cls.to(cpu)
                    reg = reg.to(cpu)
                    iou = iou.to(cpu)
                
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
                    
                    # The positive filtered labels for
                    # this FPN level. These will change based
                    # on how good each prediction is in SimOTA
                    reg_labels = torch.zeros(reg_p.shape[:-1], device=self.device)
                    
                    # The regression targets for
                    # this FPN level which will change in SimOTA
                    reg_targs = torch.negative(torch.ones(reg_p.shape, device=self.device))
                    
                    # The ground truth class for each anchor initialized
                    # as the background class
                    cls_targs = torch.zeros(reg_labels.shape, dtype=torch.long, requires_grad=False, device=self.device)
                    
                    
                    
                    
                    ### Positive Filtering
                    
                    # Get the stored SimOta paramters
                    q = self.SimOTA_params[0]
                    r = self.SimOTA_params[1]
                    extraCost = self.SimOTA_params[2]
                    SimOta_lambda = self.SimOTA_params[3]
                    
                    # Iterate over all images
                    for img in range(0, len(y_b)):
                        
                        # Use SimOTA to filter the anchors for that image
                        # and get the indices for the positive anchors
                        # and the ground truths for those anchors
                        pos_idx = SimOTA(y_b[img]['bbox'], y_b[img]['cls'], self.FPNPos[p].reshape(self.FPNPos[p].shape[0]*self.FPNPos[p].shape[1], self.FPNPos[p].shape[2]), cls_p[img], reg_p[img], q, r, extraCost, SimOta_lambda, self.losses)
                    
                        # Iterate over all positive labels and store them
                        for gt_num in range(0, len(pos_idx)):
                            
                            gt = pos_idx[gt_num]
                            
                            # Iterate over all positive labels for this gt
                            for pos in gt:
                                # Assign it as a positive lablel
                                reg_labels[img][pos] = 1
                                
                                # Assign the bounding box
                                reg_targs[img][pos] = torch.tensor(y_b[img]['bbox'][gt_num], device=self.device, requires_grad=False)
                                
                                # Assign the class
                                cls_targs[img][pos] = y_b[img]['cls'][gt_num]
                    
                    
                    
                    
                    
                    
                    ### Regression Loss 
                    
                    # The total GIoU loss
                    reg_loss = 0
                    
                    # Get the GIoU between each target bounding box
                    # and each predicted bounding box
                    
                    # Iterate over all batch elements
                    for b_num in range(0, obj_p.shape[0]):
                        # The predicted bounding boxes
                        bbox = reg_p[b_num]
                        
                        # Indices for each predicted bounding box
                        # ground truth value
                        GTs = reg_targs[b_num]
                        
                        # If there are no positive targets for this image,
                        # skip this iteration
                        if bbox[reg_labels[b_num] != 0].shape[0] == 0:
                            continue
                        
                        # Get the GIoU Loss and Value for the positive targets
                        GIoU_loss, _ = self.losses.GIoU(bbox[reg_labels[b_num] != 0], GTs[reg_labels[b_num] != 0])
                        
                        # Sum the loss across all images and save it
                        reg_loss += GIoU_loss.sum()
                    
                    
                    
                    ### Class Loss
                    
                    # One hot encode the targets
                    cls_targs_hot = nn.functional.one_hot(cls_targs, cls_p.shape[-1])
                    
                    # Get the loss value
                    cls_Loss = self.losses.FocalLoss(cls_p[reg_labels != 0], cls_targs_hot[reg_labels != 0].float()).sum()
                    
                    
                    
                    
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
                            GT_bbox = torch.tensor(y_b[b_num]["bbox"], device=self.device, requires_grad=False)
                            
                            # The best GIoU values for each predicted bounding box
                            best_GIoU = torch.negative(torch.ones(obj.shape, requires_grad=False, device=self.device, dtype=torch.float))
                            
                            # Iterate over all GT boxes
                            for box in GT_bbox:
                                
                                # Get the IoU between the predicted boxes and
                                # the current GT box
                                _, IoU_val = self.losses.IoU(reg_p[b_num], box)
                                
                                # Get the max GIoU value for each bounding box
                                # between the new GIoU values and the current
                                # saved ones. Save the max values
                                best_GIoU = torch.maximum(best_GIoU, IoU_val)
                        
                        # Get the loss between the batch elements
                        # Note: We don't just want the positively
                        # labelled ones since we want the model to learn
                        # both bad and good predictions
                        obj_Loss += self.losses.BinaryCrossEntropy(obj[reg_labels[b_num] != 0], best_GIoU[reg_labels[b_num] != 0])
                    
                    
                    
                    
                    ### Final Loss ###
                    
                    # Get the final loss for this prediction
                    N_pos = max(torch.count_nonzero(reg_labels).item(), 1)
                    finalLoss = (1/N_pos)*cls_Loss + self.reg_weight*((1/N_pos)*reg_loss) + (1/N_pos)*obj_Loss
                    totalLoss += finalLoss
                
                
                ### Updating the model
                
                # Backpropogate the loss
                totalLoss.backward()
                
                # Step the optimizer
                self.optimizer.step()
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Update the batch loss
                batchLoss += totalLoss.cpu().detach().numpy().item()
            
            
            
            # If there was somehow no positive values, then
            # skip this iteration
            if torch.where(reg_labels != 0)[0].cpu().numpy().shape[0] == 0:
                print("No positive anchors... Skipping output\n\n")
                continue
            
            # Random predictions and labels to print
            idx = np.random.choice(torch.where(reg_labels != 0)[0].cpu().numpy())
            
            
            # Display some predictions
            print(f"Step #{epoch}      Total Batch Loss: {batchLoss}")
            print("Reg:")
            print(f"Prediction:\n{reg_p[idx][reg_labels[idx] == 1][:2].cpu().detach().numpy()}")
            print(f"Ground Truth:\n{reg_targs[idx][reg_labels[idx] == 1][:2].cpu().detach().numpy()}")
            
            print()
            print("Cls:")
            print(f"Prediction: {torch.argmax(torch.sigmoid(cls_p[idx])[reg_labels[idx] == 1][:2], dim=1).cpu().detach().numpy()}")
            print(f"Ground Truth: {cls_targs[idx][reg_labels[idx] == 1][:2].cpu().detach().numpy()}")
            print()
            print("Obj:")
            print(f"Prediction:\n{1/(1+np.exp(-obj_p[idx][reg_labels[idx] == 1][:2].cpu().detach().numpy()))}")
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
    #   loadName - The name of the file to load the model from
    #   paramLoadName - The name of the file to load the model paramters from
    def loadModel(self, loadDir, loadName, paramLoadName):
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
        self.ImgDim = data['ImgDim']
        self.numCats = data['numCats']
        self.strides = data['strides']
        self.category_Ids = data["category_Ids"]
        self.JSON_Save = data
        self.FPNShapes = [self.ImgDim//self.strides[0], self.ImgDim//self.strides[1], self.ImgDim//self.strides[2]]
        self.FPNPos = [torch.tensor([[(self.strides[i]/2 + k * self.strides[i], self.strides[i]/2 + j * self.strides[i]) for k in range(0, self.FPNShapes[i])] for j in range(0, self.FPNShapes[i])], device=self.device, dtype=torch.long) for i in range(0, len(self.strides))]
    
    
    
    
    
    
    
    
    # Get predictions from the network on some images
    # Inputs:
    #   X - The inputs into the network (images to put bounding boxes on) 
    #   batchSize - The size of the batch to split the images into
    def predict(self, X, batchSize):
        # Put the model in evaluation mode
        self.eval()
        
        # If the channel dimension is not the second dimension,
        # reshape the images
        if X.shape[1] > X.shape[3]:
            X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])

        # Split the images into batches
        if batchSize != 0:
            X_batches = torch.split(X, batchSize)
        else:
            X_batches = [X]
        
        # All predictions
        preds = []
        
       # Iterate over all batches
        for b_num in range(0, len(X_batches)):
        
            # Send the inputs through the network
            preds_batch = self.forward(X_batches[b_num])
            
            # Save the predictions
            preds.append(preds_batch)
        
        
        
        # All predictions in lists
        cls_preds = []
        reg_preds = []
        obj_preds = []
        
        # Iterate over all batches to decode them
        for b_num in range(0, len(preds)):
            # The batch of outputs
            out_batch = preds[b_num]
            
            # Get the flattened class predictions
            cls_b = [out_batch[0][0].permute(0, 2, 3, 1), out_batch[0][1].permute(0, 2, 3, 1), out_batch[0][2].permute(0, 2, 3, 1)]
            for b in range(0, len(cls_b)):
                cls_b[b] = cls_b[b].reshape(cls_b[b].shape[0], cls_b[b].shape[1]*cls_b[b].shape[2], cls_b[b].shape[3])
            
            # Get the regression predictions
            reg_b = [out_batch[1][0].permute(0, 2, 3, 1), out_batch[1][1].permute(0, 2, 3, 1), out_batch[1][2].permute(0, 2, 3, 1)]
            for b in range(0, len(reg_b)):
                reg_b[b] = reg_b[b].reshape(reg_b[b].shape[0], reg_b[b].shape[1]*reg_b[b].shape[2], reg_b[b].shape[3])
            
            # Get the objectiveness predictions
            obj_b = [out_batch[2][0].permute(0, 2, 3, 1), out_batch[2][1].permute(0, 2, 3, 1), out_batch[2][2].permute(0, 2, 3, 1)]
            for b in range(0, len(obj_b)):
                obj_b[b] = obj_b[b].reshape(obj_b[b].shape[0], obj_b[b].shape[1]*obj_b[b].shape[2], obj_b[b].shape[3])
            
            
            # Decode all regression outputs and store it as a numpy array
            for b in range(0, len(reg_b)):
                reg_b[b] = self.regDecode(reg_b[b], b).cpu().numpy()
                
                # Make sure the outputs aren't larger than the input image
                reg_b[b][:, :, :2] = np.clip(reg_b[b][:, :, :2], 0, X.shape[-1])
                reg_b[b][:, :, 2:] = np.clip(reg_b[b][:, :, 2:], 0, np.inf)
                reg_b[b][:, :, 2] = np.where(reg_b[b][:, :, 0]+reg_b[b][:, :, 2] > X.shape[-1], X.shape[-1]-reg_b[b][:, :, 0], reg_b[b][:, :, 2])
                reg_b[b][:, :, 3] = np.where(reg_b[b][:, :, 1]+reg_b[b][:, :, 3] > X.shape[-1], X.shape[-1]-reg_b[b][:, :, 1], reg_b[b][:, :, 3])
            
            # Decode all class and objectiveness predictions by sending them
            # through a sigmoid function. We do this since the loss
            # function compares the sigmoid of the predictions
            # to the actual values.
            for b in range(0, len(cls_b)):
                cls_b[b] = np.argmax((1/(1+np.exp(-cls_b[b].cpu().numpy()))), axis=-1)
                obj_b[b] = (1/(1+np.exp(-obj_b[b].cpu().numpy())))
                
            
            # Save the cls, reg, and obj predictions
            cls_preds.append(cls_b)
            reg_preds.append(reg_b)
            obj_preds.append(obj_b)
        
        
        # Delete the predictions as a torch tensor
        del preds
        
        
        # Final predictions for each input item
        cls_preds_f = [[] for i in range(len(X))]
        reg_preds_f = [[] for i in range(len(X))]
        obj_preds_f = [[] for i in range(len(X))]
        
        
        # Iterate over all batches
        for b_num in range(len(cls_preds)):
            # Iterate over all FPN levels
            for level in range(len(cls_preds[b_num])):
                # Iterate over all batch elements
                for el in range(len(cls_preds[b_num][level])):
                    # Get the cls, reg, and obj predictions
                    cls_p = cls_preds[b_num][level][el]
                    reg_p = reg_preds[b_num][level][el]
                    obj_p = obj_preds[b_num][level][el]
                    
                    # Get the mask for positive predictions
                    # Using a predefined threshold to remove
                    # clearly terrible predictions before
                    # applying nomax supression
                    mask = (obj_p > self.removal_threshold).squeeze()
                    
                    # Get the masked values
                    cls_m = cls_p[mask]
                    reg_m = reg_p[mask]
                    obj_m = obj_p[mask]
                    
                    # Ensure no negative bounding box predictions
                    reg_m = np.where(reg_m < 0, 0, reg_m)
                    
                    # Store the masked results
                    for m in range(0, cls_m.shape[0]):
                        cls_preds_f[b_num*batchSize + el].append(cls_m[m])
                        reg_preds_f[b_num*batchSize + el].append(reg_m[m])
                        obj_preds_f[b_num*batchSize + el].append(obj_m[m])
        
        # Apply nonmax supression to remove 
        # predictions that are more liekly to be correct
        reg_preds_f, obj_preds_f, cls_preds_f = soft_nonmaxSupression(reg_preds_f, obj_preds_f, cls_preds_f, self.score_thresh, self.IoU_thresh, self.losses.IoU)
                    
        # Return the predictions
        return cls_preds_f, reg_preds_f, obj_preds_f