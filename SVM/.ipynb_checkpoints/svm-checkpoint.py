import pandas as pd
import numpy as np
import glob

#data import
from dataExtraction import load_beta_data

#LIBSVM
from libsvm.svmutil import *

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

#sklearn imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_svmLight(X, y):
    mask_img = nilearn.datasets.load_mni152_brain_mask(resolution=2, threshold=0.2)
    masker = NiftiMasker(mask_img=mask_img, memory="nilearn_cache", memory_level=1).fit()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # SVM classifier
    svm_model = svm_train(y_train, X_train)
    svm_save_model('svm_beta_model', svm_model)
    y_pred, accuracy, _ = svm_predict(y_test, X_test, svm_model)
    

    # Evaluate performance
    print("Model Accuracy: " + str(accuracy))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    show()

    #print 3D model of predicted brain
    support_vectors_dict = svm_model.get_SV()
    max_feature_index = max(max(sv.keys()) for sv in support_vectors_dict)
    
    support_vectors = np.zeros((len(support_vectors_dict), max_feature_index))
    
    for i, sv in enumerate(support_vectors_dict):
        for key, value in sv.items():
            support_vectors[i, key - 1] = value 
    
    #get dual coefficients (alpha * y)
    dual_coefficients = np.array(svm_model.get_sv_coef()).flatten()
    
    #compute weight vector
    weight_1d = np.sum(support_vectors.T * dual_coefficients, axis=1)
    
    weights_3d = masker.inverse_transform(weight_1d)
    plotting.plot_stat_map(weights_3d, vmax=2, alpha=0.5, title=f"Weighted Brain")

    return svm_model

def main():
    top29Subjects = [103, 105, 106, 110, 112, 113, 115, 124, 127, 130, 
                    131, 133, 138, 142, 143, 145, 157, 159, 161, 165, 
                    173, 176, 177, 183, 187, 195, 200, 207, 208]
    X, y = load_beta_data(subjects, 'nonConfound')
    y = [1 if task == 'colorwheel' else 0 for task in y]
    train_model(X, y)