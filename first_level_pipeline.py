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


def extract_beta_weights(subject_id = None, task_type = 'colorwheel',  n_runs=1):
    """
    extract_beta_weights runs first level analysis on each of the subjects, 
    storing their weights for each task type and run in a dictionary.

    subject_id: subject ID 
    task_type: colorWheel or sameDifferent 
    n_runs: max 4 runs, use less when testing
    """
    #parameters
    tr = 2.0 

    # choose appropriate hrf model
    hrf_model = "spm + derivative"
    smoothing_fwhm = 6 # gaussian kernel width (in mm)
    drift_model = None  # cosine drift terms already in confounds
    high_pass=None,  # drift terms equivalent to high-pass filter
    n_jobs=-2,  # use all-1 available CPUs

    # choose whatever confounds you want to include
    interested_confounds = ['white_matter']

    for num_run in range(n_runs):
        preproc_path = f"../teams/a05/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_type}**run-{n_runs}**.nii.gz"
        events_path = f"../teams/a05/fmriprep/sub-{subject_id}/func/sub-{subject_id}_task-{task_type}**run-{n_runs}_events.tsv"

        #load subject nii files
        run_img = image.load_img(preproc_path)

        #load the event file for the run
        events = pd.read_csv(events_path, sep="\t", 
                            usecols=["onset", "duration"]).assign(
                            trial_type="colorwheel")

        #include confounds
        confounds = interfaces.fmriprep.load_confounds_strategy(
            preproc_path, denoise_strategy="simple")[0][interested_confounds]

        #run the first level model
        fmri_glm = FirstLevelModel(
            t_r=tr,
            hrf_model=hrf_model,
            smoothing_fwhm=smoothing_fwhm,
            drift_model=drift_model,
            minimize_memory=False   
        )

        fmri_glm = fmri_glm.fit(run_img, events, confounds)

        #design matrix = task (convolved with HRF) + confounds
        design_matrix = fmri_glm.design_matrices_[0]

        # map of parameter estimates / beta weights
        # this is the 'feature' map to use in classification
        beta_weights = fmri_glm.compute_contrast("colorwheel", output_type="effect_size")
    return beta_weights

def main():
    top29Subjects = [103, 105, 106, 110, 112, 113, 115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 187, 195, 200, 207, 208]
    taskType = ['colorWheel', 'sameDifferent']
    all_subject_features = []
    all_subject_labels = []
    
    for subjID in top29Subjects:
        for task in taskType:
            beta_weights = extract_beta_weights(subject_id=subjID, task_type=task)
            all_subject_features.append(beta_weights)
            all_subject_labels.append(task)