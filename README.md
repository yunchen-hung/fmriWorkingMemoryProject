# fmriWorkingMemoryProject (Capstone Project)

To reproduce our code, use the environment.yml to import a conda environment. This can be done with the following command: 
`conda env create -f environment.yml`. Follow these steps to reproduce our project:

1. To generate the beta weights with and without confounds, run the preprocessed fMRI data (preprocessed with fMRI prep) in `preprocessingPipelines/noConfound_first_level_pipeline.py` or `preprocessingPipelines/confound_first_level_pipeline.py`.
2. Move the saved betas into folders called `confoundBeta` or `nonConfoundBeta`
3. To retrieve the saved beta weights only within the brain itself and the task labels, `preprocessingPipelines/dataExtraction.py`.
4. Given our dataset, we can now run the model training and testing code!
   1. For SVM, run `SVM/svm.ipynb` to see the confusion matrix and coefficient brain map
   2. For Logistic Regression, run `LogisticRegression/logistic_confounds.ipynb` for the confounded betas and `LogisticRegression/logistic_nonConfounds.ipynb` for the non confounded betas, and to see the confusion matrix and coefficient brain map
   3. For Naive Bayes, run `NaiveBayes/naivebayes_confounds.ipynb` for the confounded betas and `NaiveBayes/naivebayes_nonconfounds.ipynb` for the non confounded betas
   4. For kNN, run `kNN/knn_confounds.ipynb` for the confounded betas and `kNN/knn_nonconfounds.ipynb` for the non confounded betas

Additional: To plot single subject, single run beta weights for both tasks AND the Nilearn MNI 152 brain mask we used for the mode training, run `imageResults/plotting_maps.ipynb`



#### Project Overview
In this project, we trained multiple models for multi-voxel pattern analysis (MVPA) with task fMRI. 

#### References
1. Kiyonaga, A.; Scimeca, J.; D’Esposito, M. (2018). Dissociating the causal roles of frontal and parietal cortex in working memory capacity [Registered Report Stage 1 - Protocol]. figshare. Journal contribution. [https://doi.org/10.6084/m9.figshare.7145873.v1
](https://doi.org/10.6084/m9.figshare.7145873.v1)

2. LIBSVM: Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. Software available at [http://www.csie.ntu.edu.tw/~cjlin/libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm)
    - pip install -U libsvm-official
3. Scikit-Learn: [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.