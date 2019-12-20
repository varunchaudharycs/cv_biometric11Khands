from image_features.FeatureModel import *

import cv2
from skimage.feature import hog
import task5 as lsh
import h5py
import os

CURRENT_DIR = os.path.dirname(__file__)

class HOG(FeatureModel):

    def __init__(self, ds_folder_path='../Inputs', image_label=ImageLabel.UNSPECIFIED, labelled_file=None):
        FeatureModel.__init__(self, ds_folder_path, image_label, labelled_file)

    def extract_feature_vector(self, image_id):
        folder_path = self.ds_folder_path
        image_file_path = folder_path + IMG_PREFIX + image_id + IMG_EXT
        img = cv2.imread(image_file_path)
        return self.calculate_image_hog(img)

    def calculate_image_hog(self, img):
        """
        Calculates Histogram of Gradients and returns a feature vector

        Expects an image with three color channel, BGR
        """
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # downsize the image
        yuv = cv2.resize(src = yuv, dsize=(160, 120))
        # working on the most important channel, y
        yuv = yuv[:,:,0]  # taking out only one channel
        feature = hog(yuv,
                      orientations=9,
                      pixels_per_cell=(8,8), # decreasing this increases number of features
                      cells_per_block=(2,2), # increasing this increases space usage
                      block_norm='L2-Hys',
                      visualize=False,
                      feature_vector=True,
                      multichannel=False
                     )

        return feature

    def image_hog(self, image_path):
        img = cv2.imread(image_path) # reading a colour image
        return self.calculate_image_hog(img) # all params are set inside

    def extract_feature_feature_vectors(self):
        input_images = self.get_data_set_files()
        feature_matrix = []

        # extract using code
        # for img in input_images:
        #     feature_matrix.append(self.image_hog(img.image_file_path))

        # extract from the existing features file
        lsh_index = lsh.LSHIndex()
        hdf5_file = CURRENT_DIR + os.sep + '..' + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'feature_vectors_full_data.hdf5'
        data_hog = None
        with h5py.File(hdf5_file, 'r') as hf:
            data_hog = hf['hog_features'][:]
        for i in input_images:
            img_idx = lsh_index.image_id_map[i.image_id]
            feature_vector = data_hog[img_idx]
            feature_matrix.append(feature_vector)

        return feature_matrix

    def extract_feature_feature_vectors_list(self, input_images):
        feature_matrix = []

        for img in input_images:
            feature_matrix.append(self.image_hog(img.image_file_path))

        return feature_matrix

    def find_similar_images(self, query_image_id, k):
        pass

    def find_m_similar_images(self, reduced_data_matrix, reduced_query_img_vector, m, dimensionality_reduction_model):
        # for pca use dot product similarity and for others(svd, nmf and lda) use euclidean
        if dimensionality_reduction_model == "1":
            return self.find_m_similar_images_dot(reduced_data_matrix, reduced_query_img_vector, m)
        else:
            return self.find_m_similar_images_euc(reduced_data_matrix, reduced_query_img_vector, m)
