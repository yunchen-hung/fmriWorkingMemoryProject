# fmriWorkingMemoryProject

In this project, we trained multiple models for multi-voxel pattern analysis (MVPA) with task fMRI. 

Data is sourced from Kiyonaga, A.; Scimeca, J.; D’Esposito, M. (2018). Dissociating the causal roles of frontal and parietal cortex in working memory capacity [Registered Report Stage 1 - Protocol]. figshare. Journal contribution. [https://doi.org/10.6084/m9.figshare.7145873.v1
](https://doi.org/10.6084/m9.figshare.7145873.v1)

To reproduce our code, use the environment.yml to import a conda environment. This can be done with the following command: 
conda env create -f environment.yml

The notable packages used are as follows:
1. LIBSVM: Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. Software available at [http://www.csie.ntu.edu.tw/~cjlin/libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm)
    - pip install -U libsvm-official
2. Scikit-Learn: [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.