import numpy as np







# Remove the overlapping boxes around objects to get a more
# clutter free space and the "best" predictions
# Inputs:
#   B - The predicted boudning boxes to analyze of shape (x, y, w, h)
#   S - The confidence score for each bounding box
#   C - The classes for each bounding box
#   N - The IoU threshold to remove boxes
# Outputs:
#   D - The predicted bounding boxes we want to keep
#   S - The confidence score for the predicted bounding boxes to keep
#   C - The classes for the predicted boudning boxes to keep
def soft_nonmaxSupression(B, S, C, N):
    # The bounding boxes we want to keep
    D = []
    scores = []
    classes = []
    
    # Iterate over all images
    for img in range(0, len(B)):
        # Data for the current image
        b = np.array(B[img])
        s_static = np.array(S[img]) # scores that won't we updated by the s update function
        s = np.squeeze(S[img])
        c = np.squeeze(C[img])
        
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
            
            # Iterate over every bounding box left in the
            # B array
            for bbox_num in range(0, b.shape[0]):
                bbox = b[bbox_num]
                
                ## Get the IoU between the bounding box and
                ## the bounding box with the highest confidence (M)

                # Get the (x, y) coordinates of the intersection
                xA = np.maximum(M[0], bbox[0])
                yA = np.maximum(M[1], bbox[1])
                xB = np.maximum(M[0]+M[2], bbox[0]+bbox[2])
                yB = np.maximum(M[1]+M[3], bbox[1]+bbox[3])
                
                # Get the area of the intersection
                intersectionArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
                
                # Compute the area of both rectangles
                areaA = (M[2]+1)*(M[3]+1)
                areaB = (bbox[2]+1)*(bbox[3]+1)
                
                # Get the union of the rectangles
                union = areaA + areaB - intersectionArea
                
                # Compute the intersection over union
                IoU = intersectionArea/union
                
                # Save the idx to remove the box if the IoU 
                # is greater than the threshold
                if IoU > N:
                    idx_to_remove.append(bbox_num)
                # If the IoU is not greater, update the confidence score
                else:
                    s[bbox_num] = s[bbox_num]*np.exp(-((IoU)**2)/mean_scores)
            
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