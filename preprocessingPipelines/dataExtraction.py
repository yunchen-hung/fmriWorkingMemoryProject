import pandas as pd
import numpy as np
import glob

#nilearn imports
import nilearn
from nilearn import plotting, image, interfaces, maskers
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show, plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.reporting import get_clusters_table
import nibabel as nib
from nilearn.maskers import NiftiMasker

subjects = [103, 105, 106, 109, 110, 115, 117, 124, 
           127, 130, 131, 133, 138, 140, 142, 143, 145,
           147, 157, 159, 161, 165, 172, 176, 177, 178,
           180, 181, 182, 183, 188, 200, 207, 208]

def load_beta_data(subjects, data_type):
    """
    subjects: a 1-D array of subject IDs 
    data_type: 'confound' or 'nonConfound', depending on what you are training with

    This function uses the MNI brain mask to create a uniform mask for all brain images
    """
    taskType = ['colorwheel', 'samedifferent']
    num_runs = [1, 2, 3, 4]
    
    X, y = [], []

    for subjID in subjects:
        for task in taskType:
            for run in num_runs:
                try:
                    file_path = f"~/teams/a05/group_1_data/{data_type}Beta/beta_{subjID}_{task}_{run}.nii.gz"
                    
                    beta_img = nib.load(str(file_path))
                    
                    # getting the data as an array, then flattening to 1D feature vector for model training
                    mask_img = nilearn.datasets.load_mni152_brain_mask(resolution=2, threshold=0.2)
                    masker = NiftiMasker(mask_img=mask_img, memory="nilearn_cache", memory_level=1).fit()

                    #transform the subject data to get the brain voxels
                    beta_data = masker.transform(beta_img).flatten()
                
                    X.append(beta_data)
    
                    # appending the task category to y
                    y.append(task)
                except:
                    #if there isn't a specific run, i.e. run 4
                    continue

    return np.array(X), y