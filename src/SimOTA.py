import torch
import numpy as np





# Given some object, convert it to a numpy array
def convertNumpy(item):
    if type(item) != np.array:
        if type(item) == type(torch.tensor([])):
            return item.cpu().detach().numpy()
        else:
            return np.array(item)
    return item





# Get the center prior between a gt and the anchor locations
# on the image
# Inputs:
#   A - All anchors for a single image
#   G_box - A ground truth box for this image
#   r - radius used to select anchors in this function
#   extraCost - The extra cost to add to those anchors not in
#               the r**2 radius
# Output:
#   Array with the same number of values as the number of anchors
#   where each value is the center prior value of that anchor
def centerPrior(A, G_box, r, extraCost):
    ## Center Prior selects the r**2 closest anchors according to the
    ## center distance between the anchors and gts. Those anchors
    ## that are in the radius are not subject to any extra cost, but those
    ## anchors outside the radius are assigned extra cost to avoid
    ## having them be labelled as positive anchors for this gt.
    
    # Get the center location of the ground truth boudning box
    center = (G_box[0]+(G_box[2]//2), G_box[1]+(G_box[3]//2))
    
    # Get the difference between the center locations in A and the
    # center location of the gt bounding box
    diff = A-center
    
    # Use the distance formula to get the distance from the
    # gt center location for each anchor
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Get the indices of the distances which are greater
    # than r**2 meaning the anchor is outside the radius
    idx_neg = np.where(dist > r**2)
    
    # Array of zeros corresponding to the center prior of each anchor
    c_cp = np.zeros(A.shape[0])
    
    # Those values not in the r**2 radius are subject to a constant cost
    c_cp[idx_neg] = extraCost
    
    return c_cp





# SimOTA is used for dynamic label assignment.
# Inputs:
#   G_reg - All ground truth bounding boxes in an image
#   G_cls - The class for all ground truth boxes in an image
#   A - All anchors (predicted bounding box locations) for this image
#   P_cls - The class predictions for this image
#   P_reg - The regressin predictions for this image
#   q - The number of GIoU values to pick when calculating the k values
#       - k = The number of labels (supply) each gt has
#   r - The radius used to calculate the center prior
#   extraCost - The extra cost used in the center prior computation
#   Lambda - balancing factor for the foreground loss
#   LossFuncts - A LossFunctions object used to get loss values in this function
# Output:
#   List of lists where the first dimension is the same as the number
#   of ground truths and the second is the anchors corresponding
#   to that gt.
def SimOTA(G_reg, G_cls, A, P_cls, P_reg, q, r, extraCost, Lambda, LossFuncts):
    # Note: gt = ground truth
    
    # Convert the objects to numpy arrays
    G_reg = convertNumpy(G_reg)
    G_cls = convertNumpy(G_cls)
    A = convertNumpy(A)
    P_reg = convertNumpy(P_reg)
    P_cls = convertNumpy(P_cls)
    
    # 1: Get the number of ground truths and number of anchors
    m = len(G_reg) # Number of gts
    n = A.shape[0] # Number of anchors
    
    # 2: Already have the predictions
    
    # 3: Create the supplying vector for each gt using dynamic k estimation
    # Each gt gets the sum of the top q IoU values.
    # This method is known as dynamic k estimation
    # Note: Each k represents the number of labels each gt gets
    
    # The supplying vector
    s_i = np.ones(m+1, dtype=np.int16)
    
    # The sum of all k values
    k_sum = 0
    
    
    
    
    # Iterate over all ground truth boxes (i = gt_i)
    for i in range(0, m):
        
        # Get the ith truth value
        gt = G_reg[i]
        
        # Get the (x, y) coordinates of the intersections
        xA = np.maximum(gt[0], P_reg[:, 0])
        yA = np.maximum(gt[1], P_reg[:, 1])
        xB = np.minimum(gt[0]+gt[2], P_reg[:, 0]+P_reg[:, 2])
        yB = np.minimum(gt[1]+gt[3], P_reg[:, 1]+P_reg[:, 3])
        
        # Get the area of the intersections
        intersectionArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        
        # Compute the area of both rectangles
        areaA = (gt[2]+1)*(gt[3]+1)
        areaB = (P_reg[:, 2]+1)*(P_reg[:, 3]+1)
        
        # Get the union of the rectangles
        union = areaA + areaB - intersectionArea
        
        # Compute the intersection over union for all anchors
        IoU = intersectionArea/union
        
        # Those with an IoU of 0 need to be transformed to 0
        IoU[union == 0] = 0
        IoU[np.isnan(IoU)] = 0
        
        # Get the q top IoU values (the top q predictions)
        # and sum them up to get the k for this gt
        k = np.sort(IoU)[-q:].sum()
        
        # Add the k value to the total k sum
        k_sum += k
        
        # Save the k value to the supplying vector
        # as an iteger
        s_i[i] = int(round(k))
    
    # 4: s_m+1, the background class takes all the rest of the labels
    # k_sum is essentially k*m
    s_i[-1] = n-round(k_sum)
    
    # 5: Create the demanding vector. Each anchor demands
    # 1 unit of supply
    d_j = torch.ones(n)
    
    # 6: Focal loss between the predicted classes and the actual classes
    # 7: GIoU loss between the predicted bounding boxes and the gt boxes
    # 8: Center Prior cost between the anchors and the ground truth locations
    
    # Initialize the class cost, the regression cost, and the center prior cost
    c_cls = np.zeros((m, n))
    c_reg = np.zeros((m, n))
    c_cp = np.zeros((m, n))
    
    # Get the predicted classes (instead of the probabilities)
    preds_cls = torch.argmax(torch.tensor(P_cls), dim=-1)
    
    # Iterate over all ground truth boxes (i = gt_i)
    for i in range(0, m):
        # Get the Focal loss between all predicted classes and
        # the ith ground truth class
        loss_cls = LossFuncts.FocalLoss(preds_cls, torch.tensor(G_cls[i].astype(np.int32)))
        
        # Get the GIoU loss between all predicted boudning boxes
        # and the ith ground truth box
        loss_giou = LossFuncts.GIoU(torch.tensor(P_reg), torch.tensor(G_reg[i:i+1].astype(np.int32)))[0]
        
        # Get the center prior cost between the anchor locations and the gt bounding
        # box.
        loss_cp = centerPrior(A, G_reg[i], r, extraCost)
        
        # Save the costs
        c_cls[i] = loss_cls
        c_reg[i] = loss_giou
        c_cp[i] = loss_cp
    
    # 9: Get the cost for the background class
    
    # The background class is 0, so use 0 to compare the class
    # predictions to the background using Focal Loss
    c_bg = LossFuncts.FocalLoss(preds_cls, torch.zeros(preds_cls.shape)).cpu().detach().numpy()
    c_bg = np.array([c_bg])
    
    # 10: Get the foreground cost which is the addition of
    # all costs but the background loss
    c_fg = c_cls + Lambda*c_reg + c_cp
    
    # 11: Compute the final cost matrix. Below are the shapes of each matrix:
    #  c_fg = (Number of gt, FPN Dim)
    #  c_bg = (1, FPN Dim)
    #  cost = (Number of gt + 1, FPN Dim)
    cost = np.concatenate((c_fg, c_bg))
    
    
    
    
    
    
    
    ## SimOTA:
    ## From here, SimOTA diverges from OTA by selecting the top k
    ## predictions with the least cost as positive samples. Note
    ## that the k value was calculated and put in the s_i vector
    
    # The positive indices for eahc gt
    pos_idx = []
    
    # Select the top k predictions for each gt
    for i in range(0, m):
        ## Select the top k predictions. If the value is too large,
        ## it is outside of the radius and should be counted as
        ## a negative, not a positive
        
        # Get the k value (supply) for this gt
        k = s_i[i]
        
        # The cost for this gt
        cost_gt = cost[i]
        
        # Get the indices of the top k lowest costs
        idx = np.argsort(cost_gt)[:k]
        idx = np.append(idx, 0)
        
        # If any of the costs are higher than the extra cost
        # of the background, remove it
        #idx_rem = np.where(cost_gt[idx] > extraCost)
        #idx = np.delete(idx, idx_rem)
        
        # Store the positive indices
        pos_idx.append(idx)
    
    # Return the positive indices
    return pos_idx