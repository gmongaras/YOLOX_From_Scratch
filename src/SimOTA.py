






# SimOTA is used for dynamic label assignment.
# Inputs:

def SimOTA(ground_truths, reg_predictions, OTA_lambda):
    # i = the ith ground truth box
    # j = the jth prediction
    # c_ij is a 2-d matrix where i is D1 and j is D2
    
    
    #   c_ij = cost for each gt-prediction pair.
    #      i = The ground truth boundary boxes for the image
    #      j = The predicted boundary boxes
    
    print()