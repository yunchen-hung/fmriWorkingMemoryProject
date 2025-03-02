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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from first_level_pipeline import extract_beta_weights


def train_evaluate_model(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naïve Bayes classifier
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: " + str(accuracy))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    show()

    return accuracy

def main():
    subjects = [103, 105, 106, 109, 110, 115, 117, 124, 
           127, 130, 131, 133, 138, 140, 142, 143, 145,
           147, 157, 159, 161, 165, 172, 176, 177, 178,
           180, 181, 182, 183, 188, 200, 207, 208]
    taskType = ['colorWheel', 'sameDifferent']
    all_subject_features = []
    all_subject_labels = []
    
    for subjID in subjects:
        for task in taskType:
            beta_weights = extract_beta_weights(subject_id=subjID, task_type=task)
            all_subject_features.append(beta_weights)
            all_subject_labels.append(0 if task == 'colorWheel' else 1)  # Convert task labels to binary

    X = np.array(all_subject_features)
    y = np.array(all_subject_labels)

    # Train and evaluate the model
    accuracy_score = train_evaluate_model(X, y)
    print(accuracy_score)

