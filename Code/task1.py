# from image_features.colormoments import ColorMoment
# from image_features.SIFT import SIFT
from image_features.HOG import HOG
# from image_features.LBP import LBP
from image_features.FeatureModel import *
from utils.cli_options import *
import ast
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import NMF
# import matplotlib.pyplot as plt
# import pandas as pd
import os
import cv2 as cv

CURRENT_DIR = os.path.dirname(__file__)
METADATA_DIR = CURRENT_DIR + os.sep + '..' + os.sep + 'Metadata'
IMG_PREFIX = '/Hand_'
IMG_EXT = '.jpg'


def dimension_reduction_similarity(fmodel, pca_dorsal, pca_palmar, new_data_matrix_entire_dorsal, new_data_matrix_entire_palmar, image_feature_entire, dim_reduction_tech, query_img_vector, dorsal_images, palmar_images, query_image_id):
    # Query images
    new_query_img_vector_dorsal = pca_dorsal.transform([query_img_vector])
    new_query_img_vector_palmar = pca_palmar.transform([query_img_vector])
    
    # calculate similarity
    similar_images_dorsal = fmodel.find_m_similar_images(new_data_matrix_entire_dorsal, new_query_img_vector_dorsal, 3, dim_reduction_tech)
    similar_images_palmar = fmodel.find_m_similar_images(new_data_matrix_entire_palmar, new_query_img_vector_palmar, 3, dim_reduction_tech)

    dorsal_score, palmar_score = 0, 0
    for image in similar_images_dorsal:
        dorsal_score += image.similarity_score

    for image in similar_images_palmar:
        palmar_score += image.similarity_score
        
    tp = 0
    if dorsal_score > palmar_score:
        print('Label: Dorsal\n')
        if query_image_id in dorsal_images:
            tp = 1
    else:
        print('Label: Palmar\n')
        if query_image_id in palmar_images:
            tp = 1

    return tp


def run_task():
    selected_model = "3" # HOG is pre-selected model
    dim_reduction_tech = "1" # PCA is pre-selected dimensionality reduction technique
    folder_path = input_data_set_folder_path()
    query_file_path = input_data_set_folder_path(type='training')
    
    # take k as input (number of latent space)
    k = choose_valueof_k()

    query_img_vector = None

    # PCA
    if dim_reduction_tech == "1":
        pca_dorsal = PCA(n_components=k)
        pca_palmar = PCA(n_components=k)
    # SVD
    elif dim_reduction_tech == "2":
        pca_dorsal = TruncatedSVD(n_components=k)
        pca_palmar = TruncatedSVD(n_components=k)
    # NMF
    elif dim_reduction_tech == "3":
        pca_dorsal = NMF(n_components=k, init='random', random_state=0)
        pca_palmar = NMF(n_components=k, init='random', random_state=0)

    if selected_model == "1":
        image_feature_entire = ColorMoment(folder_path)
        image_feature_model_dorsal = ColorMoment(folder_path, image_label=ImageLabel.DORSAL)
        image_feature_model_palmar = ColorMoment(folder_path,image_label=ImageLabel.PALMAR)

        # query_file_path = input('Enter the path to the query image: ')
        image_feature_query = ColorMoment(query_file_path)

        # input_images = image_feature_model_dorsal.get_data_set_files()
        X_entire = image_feature_entire.extract_feature_feature_vectors()
        X_entire = np.array(X_entire)

        feature_matrix_dorsal = image_feature_model_dorsal.extract_feature_feature_vectors()
        X_dorsal = np.array(feature_matrix_dorsal)

        feature_matrix_palmar = image_feature_model_palmar.extract_feature_feature_vectors()
        X_palmar = np.array(feature_matrix_palmar)

        pca_dorsal.fit(X_dorsal)
        pca_palmar.fit(X_palmar)
        # All dorsal images
        new_data_matrix_entire_dorsal = pca_dorsal.transform(X_entire)
        # All Palmar images
        new_data_matrix_entire_palmar = pca_palmar.transform(X_entire)

        # Used to calculate accuracy of the classification
        dorsal_images, palmar_images = get_dorsal_palmar_images()

        datasetfiles = image_feature_query.get_data_set_files()
        tp = 0
        for i in datasetfiles:
            print("Image ID: {:s}".format(i.image_id))
            query_image_id = i.image_id
            query_img_vector = image_feature_query.extract_feature_vector(query_image_id)
            tp += dimension_reduction_similarity(image_feature_entire, pca_dorsal, pca_palmar, new_data_matrix_entire_dorsal, new_data_matrix_entire_palmar, image_feature_entire, dim_reduction_tech, query_img_vector, dorsal_images, palmar_images, query_image_id)

        print('\nClassifier Accuracy: ' + str((tp/100)*100))


    elif selected_model == "3":
        image_feature_entire = HOG(folder_path)
        image_feature_model_dorsal = HOG(folder_path, image_label=ImageLabel.DORSAL)
        image_feature_model_palmar = HOG(folder_path,image_label=ImageLabel.PALMAR)

        # query_file_path = input('Enter the path to the query image: ')
        image_feature_query = HOG(query_file_path)

        # input_images = image_feature_model_dorsal.get_data_set_files()
        X_entire = image_feature_entire.extract_feature_feature_vectors()
        X_entire = np.array(X_entire)

        feature_matrix_dorsal = image_feature_model_dorsal.extract_feature_feature_vectors()
        X_dorsal = np.array(feature_matrix_dorsal)

        feature_matrix_palmar = image_feature_model_palmar.extract_feature_feature_vectors()
        X_palmar = np.array(feature_matrix_palmar)

        pca_dorsal.fit(X_dorsal)
        pca_palmar.fit(X_palmar)
        # All dorsal images
        new_data_matrix_entire_dorsal = pca_dorsal.transform(X_entire)
        # All Palmar images
        new_data_matrix_entire_palmar = pca_palmar.transform(X_entire)

        # Used to calculate accuracy of the classification
        dorsal_images, palmar_images = get_dorsal_palmar_images()

        datasetfiles = image_feature_query.get_data_set_files()
        tp = 0
        for i in datasetfiles:
            print("Image ID: {:s}".format(i.image_id))
            query_image_id = i.image_id
            query_img_vector = image_feature_query.extract_feature_vector(query_image_id)
            tp += dimension_reduction_similarity(image_feature_entire, pca_dorsal, pca_palmar, new_data_matrix_entire_dorsal, new_data_matrix_entire_palmar, image_feature_entire, dim_reduction_tech, query_img_vector, dorsal_images, palmar_images, query_image_id)

        print('\nClassifier Accuracy: ' + str((tp/100)*100))


if __name__ == "__main__":
    run_task()
