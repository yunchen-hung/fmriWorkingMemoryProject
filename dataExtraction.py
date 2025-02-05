import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

#nilearn imports
from nilearn import plotting, image, interfaces
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show, plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.reporting import get_clusters_table
import nilearn
import nibabel as nib

top29Subjects = [103, 105, 106, 110, 112, 113, 115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 187, 195, 200, 207, 208]

def load_beta_data(top29Subjects):
    taskType = ['colorwheel', 'samedifferent']
    num_runs = [1, 2, 3, 4]
    
    X, y = [], []

    for subjID in top29Subjects:
        for task in taskType:
            for run in num_runs:
                try:
                    file_path = f"~/teams/a05/group_1_data/nonConfoundBeta/beta_{subjID}_{task}_{run}.nii.gz"
    
                    # loading image
                    beta_img = nib.load(str(file_path))
    
                    # getting the data as an array, then flattening to 1D feature vector for model training
                    beta_data = beta_img.get_fdata().flatten()
                    X.append(beta_data)
    
                    # appending the task category to y
                    y.append((task))
                except:
                    #if there isn't a specific run, i.e. run 4
                    continue
    
    return np.array(X), y