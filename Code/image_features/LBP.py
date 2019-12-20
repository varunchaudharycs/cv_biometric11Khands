from image_features.FeatureModel import *
from skimage import feature
import cv2 as cv
import numpy as np
import task5 as lsh
import h5py
import os

CURRENT_DIR = os.path.dirname(__file__)

class LBP(FeatureModel):
    # Initialize the input path
    def __init__(self, ds_folder_path='../Inputs', image_label=ImageLabel.UNSPECIFIED):
        FeatureModel.__init__(self, ds_folder_path, image_label)

    def extract_feature_vector(self, image_id):
        image_path = self.get_path_from_id(image_id)
        return self.image_lbp(image_path)

    def calculate_image_lbp(self, image, radius=3):
        num_points = 8 * radius
        cell_size = 100 * 100
        interval = 100
        rows = 1200
        columns = 1600
        n_bins = 20
        count = 0
        lbp_local_max = int(2 ** num_points)
        number_of_cell = int(rows * columns / (cell_size))
        number_of_features = int(number_of_cell * n_bins)
        lbp_img = np.zeros(number_of_features)
        for i in range(0, rows, interval):
            for j in range(0, columns, interval):
                lbp_local = feature.local_binary_pattern(image[i:i + 100, j:j + 100], num_points, radius)
                hist, _ = np.histogram(lbp_local, density=False, bins=n_bins, range=(0, lbp_local_max))
                norm_val = np.sqrt(np.sum(hist ** 2) * 1.0)
                if norm_val != 0:
                    hist = hist / norm_val
                # Normalization done above
                lbp_img[count * n_bins: (count + 1) * n_bins] = hist
                count += 1
        return lbp_img

    def image_lbp(self, image_path):
        img = cv.imread(image_path, 0)
        return self.calculate_image_lbp(img, radius=2)

    def extract_feature_feature_vectors(self):
        feature_matrix = []
        input_images = self.get_data_set_files()

        # for i in input_images:
        #     feature_matrix.append(self.extract_feature_vector(i.image_id))

        # extract from the existing features file
        lsh_index = lsh.LSHIndex()
        hdf5_file = CURRENT_DIR + os.sep + '..' + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'feature_vectors_full_data.hdf5'
        data_lbp = None
        with h5py.File(hdf5_file, 'r') as hf:
            data_lbp = hf['lbp_features'][:]

        for i in input_images:
            img_idx = lsh_index.image_id_map[i.image_id]
            feature_vector = data_lbp[img_idx]
            feature_matrix.append(feature_vector)

        return feature_matrix

    def extract_feature_feature_vectors_list(self, input_images):
        feature_matrix = []
        for i in input_images:
            feature_matrix.append(self.extract_feature_vector(i.image_id))
        return feature_matrix

    def find_m_similar_images(self, reduced_data_matrix, reduced_query_img_vector, m, dimensionality_reduction_model):
        distances = []
        for i in range(len(reduced_data_matrix)):
            temp = (self._input_images_[i],
                    self.euclidean_distance(list(reduced_data_matrix[i]), list(reduced_query_img_vector[0])))
            distances.append(temp)

        counter = 0
        sorted_similar_images = []
        for i in sorted(distances, key=lambda x: x[1]):
            if counter < m:
                # custom definition for similarity score
                # similarity score = (reciprocal of distance) times 1000
                # 0.000001 for LDA similarity score computation
                similarity_score = round((1 / i[1] + 0.000001) * 10, 3)
                # similarity_score = i[1]

                image_match = ImageMatch(i[0], similarity_score)
                sorted_similar_images.append(image_match)

                print(f'{counter + 1}. Image Id: {i[0].image_id}  Similarity score: {similarity_score}')
                counter += 1

        return sorted_similar_images