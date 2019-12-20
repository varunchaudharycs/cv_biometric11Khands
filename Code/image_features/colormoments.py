"""
Color Moments Class

Output:
Assuming that the the file is stored in Code/image_features folder,
Output will be stored in the "Outputs/cm/" folder where we will have image descriptors
 
Use the following code to initiate Color Moments Task 3 from Phase 1:
-------------------
# Utility function to select the Model to extract image features
selected_model = choose_feature_model()
folder_path = input('Enter folder path: ')

# take image ID as input
img_id = input('Image Id: ')

# take k as input (number of the maximum output similar images)
k = int(input('Enter the value of k:\n'))

# Color Moments Model
if selected_model == "1":
    cm = ColorMoment(folder_path)
    filepath = cm.extract_feature_feature_vectors()
-------------------

@author: Abhishek Mugal
"""

# import the required libraries and packages
from image_features.FeatureModel import *
import cv2
import ast
import scipy.stats
import statistics
import glob
import os.path
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
import task5 as lsh
import h5py

# Global Variables
CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = CURRENT_DIR + '/../../Outputs'
IMG_PREFIX = '/Hand_'
IMG_EXT = '.jpg'
CM_OUTPUT_DIR = OUTPUT_DIR + '/CM_img_feature_vector.txt'


# Color Moment (CM) Class contains all the methods related to CM
class ColorMoment(FeatureModel):
    def __init__(self, ds_folder_path='../Inputs', image_label=ImageLabel.UNSPECIFIED, labelled_file=None):
        FeatureModel.__init__(self, ds_folder_path, image_label, labelled_file)

    # function to parse directory required in Task 2 and 3
    # @params: folder path of the images, model (CM or SIFT), task (default 2)
    # @returns: for color-moments, returns file path of the newly created file
    #           for SIFT, creates separate file for each image
    def extract_feature_feature_vectors(self):
        color_moments = []

        # return value: img_id and image path in that particular folder
        input_images = self.get_data_set_files()

        # check if the file already exists or not
        # if os.path.exists(CM_OUTPUT_DIR):
        #     return CM_OUTPUT_DIR, input_images

        # extract using code
        # for i in input_images:
        #     color_moments.append(self.extract_feature_vector(i.image_id))

        # extract from the existing features file
        lsh_index = lsh.LSHIndex()
        hdf5_file = CURRENT_DIR + os.sep + '..' + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'feature_vectors_full_data.hdf5'
        data_cm = None
        with h5py.File(hdf5_file, 'r') as hf:
            data_cm = hf['cm_features'][:]
        for i in input_images:
            img_idx = lsh_index.image_id_map[i.image_id]
            feature_vector = data_cm[img_idx]
            color_moments.append(feature_vector)

        return color_moments

    def extract_feature_feature_vectors_list(self, input_images):
        color_moments = []

        # check if the file already exists or not
        # if os.path.exists(CM_OUTPUT_DIR):
        #     return CM_OUTPUT_DIR, input_images

        for i in input_images:
            color_moments.append(self.extract_feature_vector(i.image_id))

        return color_moments

        # if color_moments != None:
        #     # return a common image descriptor file path
        #     return self.write_color_moments(color_moments, 2), input_images

    # function to get color moments image descriptors
    # @params: image ID, folder path of the image
    # @returns: dictionary formatted color moments with image ID
    def extract_feature_vector(self, img_id):

        img_path = self.ds_folder_path + IMG_PREFIX + img_id + IMG_EXT
        # get the image
        # img = cv2.imread(img_path, 0)/255.0
        img = cv2.imread(img_path, 0)

        # convert the image into YUV model
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # get the image height and width
        img_height = img.shape[0]
        img_width = img.shape[1]

        # create 100x100 blocks based on the image resolution
        X = 100
        Y = 100
        moments = []

        for i in range(0, img_height, X):
            for j in range(0, img_width, Y):
                # 100x100 image block
                img_block = img[i: i + X, j: j + Y].reshape(-1)

                mean = np.mean(img_block)
                std = np.std(img_block)
                skewness = 1.11**(scipy.stats.skew(img_block))
                moments = moments + [mean, std, skewness]

        # fmoments = {}
        # fmoments[img_id] = moments
        return moments


    # function to generate features matrix given a list of dictionary formatted data
    # @params: features in dictionary format key = img_id and value = 9 item feature vector
    # @returns: 2D matrix and negative flag value
    def get_feature_matrix(self, features):
        matrix = []
        for i in features:
            for _, value in i.items():
                matrix.append(value)

        return matrix


    def find_m_similar_images(self, reduced_data_matrix, reduced_query_img_vector, m, dimensionality_reduction_model):
        return self.find_m_similar_images_euc(reduced_data_matrix, reduced_query_img_vector, m)

    # function to write the color moments to the file depending on the selected Task
    # @params:  color moments dictionary object or list of dictonary objects
    #           folder path of the images
    #           task, by default the value is 1
    # @returns: output file path if file is created else False
    def write_color_moments(self, color_moments, task=1):
        if task == 1:
            img_id = color_moments['img_id']
            moments = color_moments['color_moments']

            outputFile = OUTPUT_DIR + '/cm/CM_img_desc_' + img_id + '.txt'
            try:
                with open(outputFile, 'w+') as fp:
                    print(moments, file=fp)

                print('Success! File created: ' + outputFile)
                fp.close()
                return outputFile

            except:
                print('Error while creating file.')
                return False

        elif task == 2:
            outputFile = CM_OUTPUT_DIR
            try:
                with open(outputFile, 'w+') as fp:
                    # print(color_moments, file=fp)
                    fp.write('[')
                    for index, item in enumerate(color_moments):
                        if index == (len(color_moments) - 1):
                            fp.write("%s" % item)
                        else:
                            fp.write("%s,\n" % item)
                    fp.write(']')

                print('Success! File created: ' + outputFile)
                fp.close()
                return outputFile

            except:
                print('Error while creating file.')
                return False
