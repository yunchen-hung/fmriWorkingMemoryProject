import pandas as pd
import numpy as np
import glob
import sys
import os
sys.path.append(os.path.abspath("../preprocessingPipelines")) 

from dataExtraction import *
from confound_first_level_pipeline import main
from noConfound_first_level_pipeline import main

#nilearn imports
from nilearn import plotting, image, interfaces
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show, plot_design_matrix
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.reporting import get_clusters_table
from nilearn.input_data import NiftiMasker

#sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def load_beta_data_nonconfound():
    
    X, y = [], []
    
    directories = [Path("~/teams/a05/group_1_data/nonConfoundBeta/").expanduser(), 
    Path("~/teams/a05/group_1_data/nonConfoundBeta/betas/").expanduser()]

    for directory in directories:
        
        beta_files = list(directory.glob("beta_*.nii.gz"))

        for file_path in beta_files:
            
            file_name = file_path.stem
            subjID = file_name.split("_")[1]
            task = file_name.split("_")[2]

            beta_img = nib.load(str(file_path))
            beta_data = beta_img.get_fdata().flatten()
            
            X.append(beta_data)
            y.append(task)

    return np.array(X), np.array(y)

def load_beta_data_confound():
    
    X, y = [], []
    
    directories = [Path("~/teams/a05/group_1_data/confoundBeta/").expanduser(), 
    Path("~/teams/a05/group_1_data/confoundBeta/betas/").expanduser()]

    for directory in directories:
        
        beta_files = list(directory.glob("beta_*.nii.gz"))

        for file_path in beta_files:
            
            file_name = file_path.stem
            subjID = file_name.split("_")[1]
            task = file_name.split("_")[2]

            beta_img = nib.load(str(file_path))
            beta_data = beta_img.get_fdata().flatten()
            
            X.append(beta_data)
            y.append(task)

    return np.array(X), np.array(y)

def split_train_test_valid(X, y): #split data into 20% test 20% validation 60% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) #0.25 * 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test

def KNN_nonconfound(X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier(n_neighbors=12)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return model

def KNN_confound(X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier(n_neighbors=19)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return model


def main():
    # Load non-confound data
    X_nonconfound, y_nonconfound = load_beta_data_nonconfound()
    X_train_nonconfound, X_val_nonconfound, X_test_nonconfound, y_train_nonconfound, y_val_nonconfound, y_test_nonconfound = split_train_test_valid(X_nonconfound, y_nonconfound)
    
    # Load confound data
    X_confound, y_confound = load_beta_data_confound()
    X_train_confound, X_val_confound, X_test_confound, y_train_confound, y_val_confound, y_test_confound = split_train_test_valid(X_confound, y_confound)

    # Train and display results for non-confound data
    print("Results for Non-Confound Data:")
    model_nonconfound = KNN_nonconfound(X_train_nonconfound, y_train_nonconfound, X_test_nonconfound, y_test_nonconfound)

    # Train and display results for confound data
    print("Results for Confound Data:")
    model_confound = KNN_confound(X_train_confound, y_train_confound, X_test_confound, y_test_confound)

if __name__ == "__main__":
    main()