import pandas as pd
import numpy as np

#nilearn imports
from nilearn import plotting, image, interfaces
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show, plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.reporting import get_clusters_table
from nilearn.input_data import NiftiMasker

#sklearn imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from pathlib import Path
import os.path 


def extract_beta_weights(num_run, subject_id = None, task_type = 'colorwheel'):
    """
    extract_beta_weights runs first level analysis on each of the subjects, 
    storing their weights for each task type and run in a dictionary.

    subject_id: subject ID 
    task_type: colorWheel or sameDifferent 
    num_run: max 4 runs, use less when testing
    """
    #parameters
    tr = 2.0 

    # choose appropriate hrf model
    hrf_model = "glover + derivative"
    smoothing_fwhm = 6 # gaussian kernel width (in mm)
    drift_model = None  # cosine drift terms already in confounds
    high_pass=None,  # drift terms equivalent to high-pass filter
    n_jobs=-2,  # use all-1 available CPUs

    # choose whatever confounds you want to include
    # jen: included rotation and translation along all axes, 
    # and the white matter and cerebral spinal fluid
    interested_confounds = ["rot_x", "rot_y", "rot_z", 
                        "trans_x", "trans_y", "trans_z", 
                        "white_matter", "csf"]

    print(f"subj:{subject_id}, run {num_run}")
    try:
        preproc_path = f"~/teams/a05/group_1_data/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_type}_acq-multiband_run-{num_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        openVersion = os.path.expanduser(f"~/teams/a05/group_1_data/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_type}_acq-multiband_run-{num_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
        events_path = f"~/teams/a05/group_1_data/fmriprep/events/sub-{subject_id}_task-{task_type}_acq-multiband_run-{num_run}_events.tsv"
    
        #load subject nii files
        run_img = image.load_img(preproc_path)
    
        #load the event file for the run
        events = pd.read_csv(events_path, sep="\t", 
                            usecols=["onset", "duration"]).assign(
                            trial_type=task_type)

        #include confounds
        confounds = interfaces.fmriprep.load_confounds_strategy(
            openVersion, denoise_strategy="simple")[0]
        confounds = confounds[interested_confounds]


    except:
        #if there isn't a specific run, i.e. run 4
        print("no run found!")
        return

    #run the first level model
    fmri_glm = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        smoothing_fwhm=smoothing_fwhm,
        drift_model=drift_model,
        minimize_memory=False   
    )

    #fit the model
    print(run_img.shape)
    print(confounds.shape)
    fmri_glm = fmri_glm.fit(run_img, events, confounds)

    #design matrix = task (convolved with HRF) + confounds
    design_matrix = fmri_glm.design_matrices_[0]
    
    # map of parameter estimates / beta weights
    #this is the 'feature' map to use in classification
    beta_weights = fmri_glm.compute_contrast(task_type, output_type="effect_size")
    
    return beta_weights

def save_beta(img, subject_id = None, task_type = 'colorwheel', run_num=1):
    #save to the private folder, under a folder named "betas"
    output_dir = Path("~/private/betas/")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    img.to_filename(
        output_dir / f"beta_{subject_id}_{task_type}_{run_num}.nii.gz"
    )

def main():
    top29Subjects = [103, 105, 106, 109, 110, 
                    115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 200, 207, 208]

    testSubjects =  [#107AB, 
                    117, 
                    140,
                    147,
                    172, #173AB, 
                    178, 180, 181, 182,
                    188
                    ]
    taskType = ['colorwheel', 'samedifferent']
    num_runs = [1, 2, 3, 4]
    
    for subjID in testSubjects:
        for task in taskType:
            for run in num_runs:
                beta_weights = extract_beta_weights(run, subject_id=subjID, task_type=task)

                if beta_weights is not None:
                    #save beta weights
                    save_beta(beta_weights, subjID, task, run)

if __name__ == "__main__":
    main()

