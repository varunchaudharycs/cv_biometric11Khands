import os
from math import sqrt
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import ceil

IMG_PREFIX = '/Hand_'
IMG_EXT = '.jpg'
CURRENT_DIR = os.path.dirname(__file__)


class ImageLabel(Enum):
    UNSPECIFIED = -1
    LEFT_HAND = 1
    RIGHT_HAND = 2
    DORSAL = 3
    PALMAR = 4
    WITH_ACCESSORIES = 5
    WITHOUT_ACCESSORIES = 6
    MALE = 7
    FEMALE = 8


class TrainingImage:
    def __init__(self, image_id, image_file_path):
        self.image_id = image_id
        self.image_file_path = image_file_path


class ImageMatch:
    def __init__(self, matched_image, similarity_score):
        self.matched_image = matched_image
        self.similarity_score = similarity_score


class FeatureModel:
    # Add a new parameter for Feature Model to filter images
    def __init__(self, ds_folder_path='../Inputs', image_label=ImageLabel.UNSPECIFIED, labelled_file=None):
        self.ds_folder_path = os.path.abspath(ds_folder_path)
        self.image_label = image_label
        self.labelled_file = labelled_file
        # self.model_folder_path = model_folder_path
        self._input_images_ = []

    def get_path_from_id(self, image_id):
        return str(self.ds_folder_path + IMG_PREFIX + image_id + IMG_EXT)

    def extract_feature_vector(self, image_id):
        raise NotImplementedError("No implementation provided!")

    def extract_feature_feature_vectors(self):
        raise NotImplementedError("No implementation provided!")

    def find_similar_images(self, query_image_id):
        raise NotImplementedError("No implementation provided!")

    def find_m_similar_images(self, reduced_data_matrix, reduced_query_img_vector, m, dimensionality_reduction_model):
        raise NotImplementedError("No implementation provided!")

    def find_m_similar_images_euc(self, reduced_data_matrix, reduced_query_img_vector, m):
        distances = []
        for i in range(len(reduced_data_matrix)):
            temp = (self._input_images_[i],
                    self.euclidean_distance(list(reduced_data_matrix[i]),
                                            list(reduced_query_img_vector[0])))
            distances.append(temp)

        counter = 0
        sorted_similar_images = []
        for i in sorted(distances, key=lambda x: x[1]):
            if counter < m:
                # custom definition for similarity score
                # similarity score = (reciprocal of distance) times 1000
                # 0.000001 for LDA similarity score computation
                similarity_score = round((1 / (i[1] + 0.000001)) * 1000, 3)
                # similarity_score = i[1]

                image_match = ImageMatch(i[0], similarity_score)
                sorted_similar_images.append(image_match)

                # print(f'{counter + 1}. Image Id: {i[0].image_id}  Similarity score: {similarity_score}')
                counter += 1

        return sorted_similar_images

    def find_m_similar_images_dot(self, reduced_data_matrix, reduced_query_img_vector, m):
        distances = []
        for i in range(len(reduced_data_matrix)):
            temp = (self._input_images_[i],
                    self.dot_pdt_similarity(list(reduced_data_matrix[i]), list(reduced_query_img_vector[0]))
                    )
            distances.append(temp)

        counter = 0
        sorted_similar_images = []
        for i in sorted(distances, key=lambda x: x[1], reverse=True):
            if counter < m:
                similarity_score = i[1]

                image_match = ImageMatch(i[0], similarity_score)
                sorted_similar_images.append(image_match)

                # print(f'{counter + 1}. Image Id: {i[0].image_id}  Similarity score: {similarity_score}')
                counter += 1

        return sorted_similar_images

    def get_data_set_files(self):
        input_images = []
        label_images = []
        if self.image_label != ImageLabel.UNSPECIFIED:
            label_images = get_all_image_names_for_label(self.image_label, self.labelled_file)

        image_files = os.listdir(self.ds_folder_path)
        for filename in sorted(image_files):
            if filename.endswith(".jpg"):
                if self.image_label != ImageLabel.UNSPECIFIED and filename not in label_images:
                    continue
                file_arr = filename.split('_')[1]
                img_id = file_arr.split('.')[0]
                file_path = os.path.join(self.ds_folder_path, filename)
                input_image = TrainingImage(img_id, file_path)
                input_images.append(input_image)

        self._input_images_ = input_images
        return input_images

    def input_query_image_and_validate(self):
        # get the query image id from the user
        query_image_id = input('Enter query image ID: ')
        while not self.image_exists_in_the_training_set(query_image_id):
            print('The specified image does not exists in the training image set')
            query_image_id = input('\nEnter query image ID: ')

        return query_image_id

    # checks whether given image id exists in the training, used before searching an image
    def image_exists_in_the_training_set(self, image_id):
        for image in self._input_images_:
            if image.image_id == image_id:
                return True
        return False

    # Calculate euclidean distance
    @staticmethod
    def euclidean_distance(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.sqrt(np.sum((v1-v2*1.0)**2))
        # v1 and v2 vector length is the same
#         sq_sum = 0
#         for i in range(0, len(v1)):
#             sq_sum = sq_sum + ((v2[i] - v1[i]) * (v2[i] - v1[i]))
#         return sqrt(sq_sum)

    @staticmethod
    def cosine_similarity(x, y):
        x = np.array(x)
        y = np.array(y)
        return (np.dot(x,y))*1.0/((np.sum(x*x)*np.sum(y*y))**0.5)

    @staticmethod
    def dot_pdt_similarity(x, y):
        x = np.array(x)
        y = np.array(y)
        return (np.dot(x,y))*1.0

    @staticmethod
    def chi_squared_distance(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        filt = np.logical_and(x != 0, y != 0)
        return 0.5*np.sum(1.0*((x[filt]-y[filt])**2)/(x[filt]+y[filt]))

    # function to display similar images
    # @params:  similar_images - list of tuple of similar images obtained from calculate_similarity function
    #           k - number of components provided by the user
    # @return:  void
    @staticmethod
    def display_similar_images(similar_images: [ImageMatch], k: int):
        num_of_rows_in_plot_grid = int(ceil(k / 3))
        num_of_cols_in_plot_grid = 4
        f, axes = plt.subplots(num_of_rows_in_plot_grid, num_of_cols_in_plot_grid, squeeze=False, figsize=(14, 7))

        title = 'Similar Images'
        f.canvas.set_window_title(title)

        counter = 0
        for i in range(num_of_rows_in_plot_grid):
            for j in range(num_of_cols_in_plot_grid):
                if counter >= k:
                    axes[i][j].axis('off')
                    continue
                else:
                    if counter < len(similar_images):
                        img_path = similar_images[counter].matched_image.image_file_path
                        img_id = similar_images[counter].matched_image.image_id
                        img_bgr = cv.imread(img_path)
                        img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                        axes[i][j].imshow(img)
                        axes[i][j].set_title('Img Id: %s\nScore: %f' % (img_id, similar_images[counter].similarity_score))
                        axes[i][j].axis('off')
                    else:
                        axes[i][j].axis('off')
                counter += 1
        plt.show()


def get_all_image_names_for_label(image_label, labelled_file=None):
    if labelled_file != None:
        data_frame = pd.read_csv(labelled_file)
    else:
        data_frame = pd.read_csv(os.path.join(CURRENT_DIR, '..' + os.sep, '..' + os.sep, 'Metadata', 'HandInfo.csv'))

    image_name_column = 'imageName'
    aspect_of_hand = 'aspectOfHand'
    gender_column = 'gender'
    accessories_column = 'accessories'
    dorsal_right = 'dorsal right'
    dorsal_left = 'dorsal left'
    palmar_right = 'palmar right'
    palmar_left = 'palmar left'
    male = 'male'
    female = 'female'
    with_accessories = 1
    without_accessories = 0
    images_name = []

    if image_label == ImageLabel.LEFT_HAND:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[aspect_of_hand] == dorsal_left or row[aspect_of_hand] == palmar_left:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.RIGHT_HAND:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[aspect_of_hand] == dorsal_right or row[aspect_of_hand] == palmar_right:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.DORSAL:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[aspect_of_hand] == dorsal_right or row[aspect_of_hand] == dorsal_left:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.PALMAR:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[aspect_of_hand] == palmar_right or row[aspect_of_hand] == palmar_left:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.WITH_ACCESSORIES:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[accessories_column] == with_accessories:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.WITHOUT_ACCESSORIES:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[accessories_column] == without_accessories:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.MALE:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[gender_column] == male:
                images_name.append(row[image_name_column])
    elif image_label == ImageLabel.FEMALE:
        for i in range(data_frame.shape[0]):
            row = data_frame.iloc[i]
            if row[gender_column] == female:
                images_name.append(row[image_name_column])

    return images_name

