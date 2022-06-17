import numpy as np
from torch import tensor







# Remove the overlapping boxes around objects to get a more
# clutter free space and the "best" predictions
# Inputs:
#   B - The predicted bounding boxes to analyze of shape (x, y, w, h)
#   S - The confidence score for each bounding box
#   C - The classes for each bounding box
#   score_thresh - The score threshold to remove boxes. If the score is
#                  less than this value, remove it
#   IoU_thresh - The IoU threshold to update scores. If the IoU is
#                greater than this value, update it's score
#   IoU_function - Function to calculate the IoU
# Outputs:
#   D - The predicted bounding boxes we want to keep
#   scores - The confidence score for the predicted bounding boxes to keep
#   classes - The classes for the predicted boudning boxes to keep
def soft_nonmaxSupression(B, S, C, score_thresh, IoU_thresh, IoU_function):
    # The bounding boxes we want to keep
    D = []
    scores = []
    classes = []
    
    # Iterate over all images
    for img in range(0, len(B)):
        # Data for the current image
        b = np.array(B[img])
        s_static = np.array(S[img]) # scores that won't we updated by the s update function
        s = S[img]
        c = C[img]
        
        # The bounding boxes we want to keep for this image
        D_img = []
        scores_img = []
        classes_img = []
        
        # Iterate while the predicted bounding box set isn't empty
        while b.shape[0] > 0:
        
            # Get the location of the bounding box with the highest confidence
            m = np.argmax(s)
        
            # Get the bounding box with the highest confidence
            M = b[m]
            
            # Remove the value with the highest score from the
            # predictions array (B) and add it to the
            # values we want to keep (D)
            D_img.append(M)
            scores_img.append(s_static[m])
            classes_img.append(c[m])
            b = np.delete(b, m, axis=0)
            s_static = np.delete(s_static, m, axis=0)
            s = np.delete(s, m, axis=0)
            c = np.delete(c, m, axis=0)
            
            # If there are no more elements, break the loop
            if b.shape[0] == 0:
                break
            
            # Get the mean of all cofidence scores
            mean_scores = np.mean(s)
            
            # The indices to remove after iterating over every box
            idx_to_remove = []
            
            # Get the IoU between the GT (M) and all bounding
            # boxes (b)
            _, IoU = IoU_function(tensor(M), tensor(b))
            IoU = IoU.numpy()
            
            # If the IoU is greater than the set threshold, update
            # the score of that bounding box
            idxs = np.argwhere(IoU > IoU_thresh)
            if s.shape[-1] == 1 and len(s.shape) > 1:
                s = s.squeeze(-1)
            #s[idxs] = s[idxs]*np.exp(-((IoU[idxs])**2)/mean_scores)
            s = s*np.exp(-((IoU)**2)/mean_scores)
            #s[idxs] = s[idxs]*(1-IoU[idxs])
            
            # Save the idx of the box to remove if the new score 
            # is less than the threshold
            idx_to_remove = np.squeeze(np.argwhere(s < score_thresh))
            
            # Remove the specified indices
            b = np.delete(b, idx_to_remove, axis=0)
            s_static = np.delete(s_static, idx_to_remove, axis=0)
            s = np.delete(s, idx_to_remove, axis=0)
            c = np.delete(c, idx_to_remove, axis=0)
            
        # Store the final bounding box infomration for this image
        D.append(D_img)
        scores.append(scores_img)
        classes.append(classes_img)
    
    # Return the bounding box information for all images
    return D, scores, classes