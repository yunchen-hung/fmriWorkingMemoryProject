import pandas as pd
import numpy as np

#nilearn imports
from nilearn import plotting, image
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


def extract_beta_weights(subject_id = None, task_type = 'colorWheel',  n_runs=1):
    """
    extract_beta_weights runs first level analysis on each of the subjects, 
    storing their weights for each task type and run in a dictionary.

    subject_id: subject ID 
    task_type: colorWheel or sameDifferent 
    n_runs: max 4 runs, use less when testing
    """
    #parameters
    tr = 2.0 
    hrf_model = "spm + derivative"
    smoothing_fwhm = 0.6
    drift_model = "cosine"
    run_betas = {
        'colorWheel': {},
        'sameDifferent': {}
    }

    for num_run in range(n_runs):
        #load subject nii files
        run_img = image.load_img(f'/Volumes/kiyonaga/MIPS/fMRI/subjects/MIPS_{subject_id}/func/run0{num_run+1}/srraf**.nii')

        #load the event file for the run
        events = pd.read_table(f'/Volumes/kiyonaga/MIPS/BIDS/sub-{subject_id}/func/sub-{subject_id}_task-{task_type.lower()}_acq-norm_run-{num_run+1}_events.tsv')

        #run the first level model
        fmri_glm = FirstLevelModel(
            t_r=tr,
            hrf_model=hrf_model,
            smoothing_fwhm=smoothing_fwhm,
            drift_model=drift_model,
            minimize_memory=False   
        )

        fmri_glm = fmri_glm.fit(run_img, events)

        #create design matrix
        design_matrix = fmri_glm.design_matrices_[0]

        #
        n_regressors = design_matrix.shape[1]
        activation = np.zeros(n_regressors)
        activation[0] = 1

        #should we remove confound with high_variance_confounds?

        # z-scale 
        z_map = fmri_glm.compute_contrast(activation, output_type="z_score")
       
        run_betas[task_type][f'run_{num_run+1}'] = {z_map}

        # extracting relevant features

        all_features = []
        all_labels = []

        masker = NiftiMasker(mask_img=run_img, standardize=True)
        features = masker.fit_transform(z_map)
        all_features.append(features.ravel())
        all_labels.append(0 if task_type == 'colorWheel' else 1)
    
    #return clusters_df
    return all_features, all_labels

def train_model(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM classifier
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: " + str(accuracy))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    show()

    return svm_model

def main():
    top29Subjects = [103, 105, 106, 110, 112, 113, 115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 187, 195, 200, 207, 208]
    taskType = ['colorWheel', 'sameDifferent']
    all_subject_features = []
    all_subject_labels = []
    
    for subjID in top29Subjects:
        for task in taskType:
            features, labels = extract_beta_weights(subject_id=subjID, task_type=task)
            all_subject_features.extend(features)
            all_subject_labels.extend(labels)

    X = np.array(all_subject_features)
    y = np.array(all_subject_labels)

    # Train and evaluate the model
    model = train_and_evaluate_model(X, y)