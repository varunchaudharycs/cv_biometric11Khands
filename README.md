# biometric_11Khands
Biometrix identification on 11K hands data set using opencv

CSE 515 - Multimedia and Web Databases

Project folder structure-
    - ./Code(contains Python source code for the tasks)
        - ./image_features
            - colormoments.py (Implementation to extract Color Moments features)
            - FeatureModel.py (Base class for different image feature model)
            - HOG.py (Implementation to extract HOG features)
            - LBP.py (Implementation to extract LBP features)
            - SIFT.py (Implementation for SIFT feature extraction)
        - ./utils
            - cli_options.py (common util functions to handle user inputs in the command line interface)
            - distance_measures.py - (code to calculate euclidean and chi-square distance)
            - similarity_measures.py - (code to calculate dot product and cosine similarity)
        - task1.py (Implementation for the task 1) ['dorsal' vs 'palmar' using latent semantics]
        - task2.py (Implementation for the task 2) ['dorsal' vs 'palmar' using K-Means clustering]
        - task3.py (Implementation for the task 3) ['dorsal' vs 'palmar' using Personalized Page Rank]
        - task4.py (Implementation for the task 4) ['dorsal' vs 'palmar' using classification]
        - task5.py (Implementation for the task 5) ['dorsal' vs 'palmar' using LSH]
        - task6.py (Implementation for the task 6) ['dorsal' vs 'palmar' using user feedback indexing]

    - ./Inputs(folder with test image set)
        - All the test images are stored in this folder and used while constructing the feature matrix.
    - ./Data - stores index structure or extracted features for the data set, so that it can be reused in other tasks.

    - ./Metadata(folder which contains meta data for the image data set)
        - HandInfo.csv - contains metadata for the data set
        - feature_vectors_full_data.hdf5 - stores the extracted feature vectors
        - bovw.pkl - Pickle file to store bag of visual words extracted over 11k hand images

    - ./Outputs(stores intermediate output files to be used in related next task or the actual outputs)

Libraries used -
    - scikit-learn - 0.21.3 (Decompositions like PCA, SVD, NMF, and LDA)
    - scikit-image - 0.16.1 (HOG feature extraction)
    - OpenCV-Python - 3.4.2.16 (Image loading and processing)
    - NumPy - 1.17.2 (Processing feature vectors efficiently)
    - OpenCV-contrib-Python - 3.4.2.16 (SIFT feature extraction)
    - SciPy - 1.3.1 (Color Moments feature computations, Sparse graph creation for Personalized Page Rank)
    - matplotlib -3.1.1 (Plot/draw the matched images and other visualizer consisting of images)
    - pandas - 0.25.1 (Read HandInfo.csv and other csv files as a data frame)
    - vivagraph.js - 0.12.0 (for cluster visualization)
    - h5py - 2.10.0 - to store and fetch extracted feature vector for the image
    - cvxopt - 1.2.3 - Convex optimization package - to solve optimization for SVM

Operating System - macOS
Python Version - 3.7.3
Development IDE - PyCharm, Visual Studio Code

Each tasks can be run using the respective task implementation program like below.
    python3 task1.py
Each task will prompt the user for a few required inputs params that are needed as part of the task implementation.
