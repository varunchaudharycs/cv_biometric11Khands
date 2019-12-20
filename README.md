# biometric_11Khands
Biometrix identification on 11K hands data set using opencv

CSE 515 - Multimedia and Web Databases<br />

!.[Sample data](https://imgur.com/bNTC0IY)<br />
Project folder structure-<br />
    - ./Code(contains Python source code for the tasks)<br />
        - ./image_features<br />
            - colormoments.py (Implementation to extract Color Moments features)<br />
            - FeatureModel.py (Base class for different image feature model)<br />
            - HOG.py (Implementation to extract HOG features)<br />
            - LBP.py (Implementation to extract LBP features)<br />
            - SIFT.py (Implementation for SIFT feature extraction)<br />
        - ./utils<br />
            - cli_options.py (common util functions to handle user inputs in the command line interface)<br />
            - distance_measures.py - (code to calculate euclidean and chi-square distance)<br />
            - similarity_measures.py - (code to calculate dot product and cosine similarity)<br />
        - task1.py (Implementation for the task 1) ['dorsal' vs 'palmar' using **latent semantics**]<br />
        - task2.py (Implementation for the task 2) ['dorsal' vs 'palmar' using **K-Means clustering**]<br />
        - task3.py (Implementation for the task 3) ['dorsal' vs 'palmar' using **Personalized Page Rank**]<br />
        - task4.py (Implementation for the task 4) ['dorsal' vs 'palmar' using **classification**]<br />
        - task5.py (Implementation for the task 5) ['dorsal' vs 'palmar' using **LSH**]<br />
        - task6.py (Implementation for the task 6) ['dorsal' vs 'palmar' using **user feedback indexing**]<br />
    - ./Inputs(folder with test image set)<br />
        - All the test images are stored in this folder and used while constructing the feature matrix.<br />
    - ./Data - stores index structure or extracted features for the data set, so that it can be reused in other tasks.<br />
    - ./Metadata(folder which contains meta data for the image data set)<br />
        - HandInfo.csv - contains metadata for the data set<br />
        - feature_vectors_full_data.hdf5 - stores the extracted feature vectors<br />
        - bovw.pkl - Pickle file to store bag of visual words extracted over 11k hand images<br />
    - ./Outputs(stores intermediate output files to be used in related next task or the actual outputs)<br />

Libraries used -<br />
    - scikit-learn - 0.21.3 (Decompositions like PCA, SVD, NMF, and LDA)<br />
    - scikit-image - 0.16.1 (HOG feature extraction)<br />
    - OpenCV-Python - 3.4.2.16 (Image loading and processing)<br />
    - NumPy - 1.17.2 (Processing feature vectors efficiently)<br />
    - OpenCV-contrib-Python - 3.4.2.16 (SIFT feature extraction)<br />
    - SciPy - 1.3.1 (Color Moments feature computations, Sparse graph creation for Personalized Page Rank)<br />
    - matplotlib -3.1.1 (Plot/draw the matched images and other visualizer consisting of images)<br />
    - pandas - 0.25.1 (Read HandInfo.csv and other csv files as a data frame)<br />
    - vivagraph.js - 0.12.0 (for cluster visualization)<br />
    - h5py - 2.10.0 - to store and fetch extracted feature vector for the image<br />
    - cvxopt - 1.2.3 - Convex optimization package - to solve optimization for SVM<br />

Operating System - macOS<br />
Python Version - 3.7.3<br />
Development IDE - PyCharm, Visual Studio Code<br />

Each tasks can be run using the respective task implementation program like below.<br />
    python3 task1.py<br />
Each task will prompt the user for a few required inputs params that are needed as part of the task implementation.<br />
