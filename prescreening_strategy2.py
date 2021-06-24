from numpy.lib.arraysetops import unique
import rectangle as rect
import h5py
import torch
import random
import numpy as np
import os 
from rectangle.model.networks import DenseNet as DenseNet 

from torch.utils.data import DataLoader 
import tensorflow as tf 

def standardise(image):

    batch_ = image.shape[0]
    for batch_iter_ in range(batch_):
        image[batch_iter_,...] = (image[batch_iter_,...] - \
                                torch.mean(image[batch_iter_,...]) / \
                                torch.std(image[batch_iter_,...]))

    return image

def dice_score2(y_pred, y_true, eps=1e-8):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    #y_pred[y_pred < 0.5] = 0.
    #y_pred[y_pred > 0] = 1.
    
    #Calculate the number of incorrectly labelled pixels 

    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(numerator / denominator)

def dice_fp(y_pred, y_true, pos_frames, neg_frames): 
    """ A function that computes dice score on positive frames, 
    and FP pixels on negative frames, based off Yipeng's metrics
    """
    dice_ = dice_score2(y_pred[pos_frames, :, :], y_true[pos_frames, :, :])
    fp = torch.sum(y_pred[neg_frames, :, :], dim = [1,2,3])

    return dice_, fp 
    

use_cuda = torch.cuda.is_available()

### Loading ensemble segmentation network ### 
num_ensemble = 5 
path_str = '/Users/iani/Documents/Segmentation_project/ensemble/'
latest_model = ['13.pth', '4.pth', '30.pth', '28.pth', '28.pth'] #Checked manually 
model_paths = [os.path.join(path_str, 'model_'+ str(idx), latest_model[idx]) 
for idx in range(num_ensemble)]

depth = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seg_models = [rect.model.networks.UNet(n_layers=depth, device=device,
                                    gate=None) for e in range(int(num_ensemble))]

for n, m in enumerate(seg_models):
    m.load_state_dict(torch.load(model_paths[n], map_location= device))

### Loading classifier network ### 
class_model = torch.load("/Users/iani/Documents/Segmentation_project/classification_model", map_location = device)

### Inference ###

test_file = h5py.File('/Users/iani/Documents/Reg2Seg/dataset/test.h5', 'r')
test_DS = rect.utils.io.H5DataLoader(test_file)
test_DL = DataLoader(test_DS, batch_size = 8, shuffle = False)

segmentation_threshold = 0.5 
classification_threshold = 0.5 


all_dice_screen = []
all_dice_noscreen = []

all_fp_screen = []
all_fp_noscreen = [] 

with torch.no_grad(): 

    for jj, (images_test, labels_test) in enumerate(test_DL):
        
        if use_cuda: 
            images_test, labels_test = images_test.cuda(), labels_test.cuda()

        #Obtain positive and negative frames 
        positive_frames = [(1 in label) for label in labels_test]
        negative_frames = [not(1 in label) for label in labels_test]

        #False positives negative frames

        #Dice score : positive frames 

        #Obtain prediction for classifier 
        class_preds = class_model(images_test)

        #Normalise images for segmentation network 
        norm_images_test = standardise(images_test)

        #Obtain predictions for each ensemble model and combine them
        combined_predictions = torch.zeros_like(labels_test, dtype = float)
        majority = len(seg_models) - 1 

        for model_ in seg_models:
            #Obtain predictions
            model_.eval()
            seg_predictions = torch.tensor(model_(norm_images_test) > 0.5, dtype = float)
            combined_predictions += seg_predictions

        #All segmentation results - only on positive frames 
        combined_predictions = (combined_predictions >= majority) #Majority vote 
        dice_noscreen, fp_noscreen = dice_fp(combined_predictions, labels_test, positive_frames, negative_frames)
        all_dice_noscreen.append(dice_noscreen)
        all_fp_noscreen.append(fp_noscreen)


        #dice_noscreen = dice_score(combined_predictions, labels_test)
        #all_dice_noscreen.append(dice_noscreen)

        #Pre-screened results only 
        prostate_idx = np.where(class_preds == 1)[0]
        #dice_screened = dice_score(combined_predictions[prostate_idx, :,:], labels_test[prostate_idx, :,:])
        
        positive_frames_screened = [positive_frames[i] for i in prostate_idx]
        negative_frames_screened = [negative_frames[i] for i in prostate_idx]

        dice_screen, fp_screen = dice_fp(combined_predictions[prostate_idx, :,:], labels_test[prostate_idx, :,:], positive_frames_screened, negative_frames_screened)
        all_dice_screen.append(dice_screen)
        all_fp_screen.append(fp_screen)

        print(f"Dice scores: Not-screened : {dice_noscreen} | Screened : {dice_screen}")
        print(f"FP scores: Not-screened : {fp_noscreen} | Screened : {fp_screen}")


#Obtaining plots of the histogram 

#Obtain all unique FP scores for screen, no screen method 
unique_fp_screen = [np.unique(fp_vals) for fp_vals in all_fp_screen if len(fp_vals) > 0]
unique_fp_screen = np.concatenate(unique_fp_screen, axis = 0)

#Obtain all unique FP scores for screen, no screen method 
unique_fp_noscreen = [np.unique(fp_vals) for fp_vals in all_fp_noscreen if len(fp_vals) > 0]
unique_fp_noscreen = np.concatenate(unique_fp_noscreen, axis = 0)

from matplotlib import pyplot as plt
plt.hist(unique_fp_noscreen, label = "noscreen")
plt.hist(unique_fp_screen, label = "screen")
plt.xlabel("Number of FP pixels per negative segmented frame")
plt.legend() 
plt.show()

print('Chicken')

