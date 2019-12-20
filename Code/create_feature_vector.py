from image_features.colormoments import ColorMoment
from image_features.SIFT import SIFT
from image_features.HOG import HOG
from image_features.LBP import LBP
from image_features.FeatureModel import *
from utils.cli_options import *
import numpy as np

import h5py
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# print("Current_dir: ",CURRENT_DIR)
root_path = CURRENT_DIR+os.sep+".."+os.sep+"Metadata"
hdf5_file = root_path + os.sep + "feature_vectors_full_data.hdf5"
hands_path = CURRENT_DIR+"/../../Hands"
bag_of_visual_words = root_path + os.sep + "bovw.pkl"


def create_clusters():
    with h5py.File(hdf5_file, 'r') as hf:
        data_sift = hf["sift_features"][:]
        data_sift_kp_starts = hf["sift_kp_starts"][:]

    kmeans = KMeans(n_clusters=150)
    kmeans.fit(data_sift)

    vectors = []
    for i in range(len(data_sift_kp_starts)):
        start = data_sift_kp_starts[i]
        if i == len(data_sift_kp_starts) - 1:
            descriptors = data_sift[start:]
        else:
            end = data_sift_kp_starts[i + 1]
            descriptors = data_sift[start:end]

        v = np.zeros(150)
        for desc in descriptors:
            pred = kmeans.predict([desc])
            v[pred] += 1
        vectors.append(v)
    vectors = np.array(vectors)

    with open(bag_of_visual_words, 'wb') as fp:
        pickle.dump(vectors, fp)

    print('Done')


def run_task():
    selected_model = choose_feature_model()
    # dim_reduction_tech = choose_dim_reduction_tech()
    folder_path = hands_path  # input_data_set_folder_path()

    # with h5py.File(hdf5_file, 'r') as hf:
    #     print("keys inside: ", hf.keys())

    if selected_model == "1":
        image_feature_model = ColorMoment(folder_path)
        input_images = image_feature_model.get_data_set_files()
        feature_matrix = image_feature_model.extract_feature_feature_vectors()
        X = np.array(feature_matrix)
        # dimension_reduction(X, k, selected_model, dim_reduction_tech, input_images)

    elif selected_model == "2":
        # image_feature_model = LocalBinaryPatterns(folder_path)
        image_feature_model = LBP(folder_path)
        input_images = image_feature_model.get_data_set_files()
        feature_matrix = image_feature_model.extract_feature_feature_vectors()
        X = np.array(feature_matrix)
        # dimension_reduction(X, k, selected_model, dim_reduction_tech, input_images)

    elif selected_model == "3":
        image_feature_model = HOG(folder_path)
        input_images = image_feature_model.get_data_set_files()
        feature_matrix = image_feature_model.extract_feature_feature_vectors()
        X = np.array(feature_matrix)
        # dimension_reduction(X, k, selected_model, dim_reduction_tech, input_images)

    elif selected_model == "4":
        with h5py.File(hdf5_file, 'a') as hf:
            print("keys inside: ", hf.keys())
            if "sift_features" not in hf.keys():
                image_feature_model = SIFT(folder_path)
                input_images = image_feature_model.get_data_set_files()
                feature_matrix, kp_start_end_indices = image_feature_model.extract_feature_feature_vectors()
                X = np.array(feature_matrix)
                kp_starts = np.zeros(len(kp_start_end_indices))
                print("shape of X: ", X.shape)
                print("kp-count type : ", type(kp_start_end_indices))
                print("kp_count: ", kp_start_end_indices)
                for i in range(len(input_images)):
                    curr_images_id = input_images[i].image_id
                    kp_starts[i] = int(kp_start_end_indices[curr_images_id][0])
                    print("image_id: ", curr_images_id)
                print("kp_start_points: ", kp_starts)
                kp_starts = np.array(kp_starts, dtype=np.int64)
                print("type of X: ", type(X))
                print("type of kp_starts: ", type(kp_starts))
                if "sift_features_smaller" in hf.keys():
                    del hf["sift_features_smaller"]
                if "sift_kp_starts_smaller" in hf.keys():
                    del hf["sift_kp_starts_smaller"]
                hf.create_dataset("sift_features", data=X)
                hf.create_dataset("sift_kp_starts", data=kp_starts)
            else:
                print("SIFT Key points already present")
        # dimension_reduction_sift(image_feature_model, X, k, selected_model, dim_reduction_tech, input_images, kp_start_end_indices)


if __name__ == "__main__":
    # run_task()
    create_clusters()