import pandas as pd
import numpy as np

#nilearn imports
from nilearn import plotting, image
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show, plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.reporting import get_clusters_table


#function to get first level weights for the runs
#possible task types include colorWheel or sameDifferent 

def multiSubject_fla(subject_id = None, task_type = 'colorWheel'):
    #parameters
    tr = 2.0 
    hrf_model = "spm + derivative"
    smoothing_fwhm = 0.6
    drift_model = "cosine"
    runs = 1

    for num_run in range(runs):
        print(num_run)
        #load subject nii files
        run01_img = image.load_img(f'/Volumes/kiyonaga/MIPS/fMRI/subjects/MIPS_{subject_id}/func/run0{num_run+1}/srraf**.nii')

        #load the event file for the run
        events = pd.read_table(f'/Volumes/kiyonaga/MIPS/fMRI/subjects/MIPS_{subject_id}/func/run0{num_run+1}/events.tsv')

        #run the first level model
        fmri_glm = FirstLevelModel(
            t_r=tr,
            hrf_model=hrf_model,
            smoothing_fwhm=smoothing_fwhm,
            drift_model=drift_model,
            minimize_memory=False   
        )

        fmri_glm = fmri_glm.fit(run01_img, events)

        #create design matrix
        design_matrix = fmri_glm.design_matrices_[0]

        #
        n_regressors = design_matrix.shape[1]
        activation = np.zeros(n_regressors)
        activation[0] = 1

        #should we remove confound with high_variance_confounds?

        # t-statistic to z-scale 
        z_map = fmri_glm.compute_contrast(activation, output_type="z_score")
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=0.05, height_control="fdr", two_sided=False
        )
        clusters_df = get_clusters_table(
            clean_map, stat_threshold=threshold
            )
    return clusters_df

def main():
    top29Subjects = [103, 105, 106, 110, 112, 113, 115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 187, 195, 200, 207, 208]
    taskType = ['colorWheel', 'sameDifferent']
    
    for subjID in top29Subjects:
        for task in taskType:
            multiSubject_fla(subject_id=subjID, task_type=)