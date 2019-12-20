from image_features.FeatureModel import *
import cv2 as cv
import numpy as np
import task5 as lsh
import h5py
import os
import pickle

CURRENT_DIR = os.path.dirname(__file__)

# Image match result class definition
class QueryMatch:
    def __init__(self, image_id, similarity_score):
        self.image_id = image_id
        self.similarity_score = similarity_score


class SIFT(FeatureModel):
    # image id vs feature vector
    # feature_dict = {}

    def __init__(self, ds_folder_path='../Inputs', image_label=ImageLabel.UNSPECIFIED):
        FeatureModel.__init__(self, ds_folder_path, image_label)

    def extract_keypoints_descriptors(self, image_id):
        # matching over all keypoints is computationally expensive so select best 100 keypoints from the image
        kp_threshold = 100
        kp_matrix = []
        image_file_path = self.get_path_from_id(image_id)
        img = cv.imread(image_file_path)
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(kp_threshold)
        key_points, descriptors = sift.detectAndCompute(grayscale_img, None)

        kp_count = len(key_points)
        for i in range(0, kp_count):
            # Can include keypoint location, angle as well to the feature vector
            # kp = key_points[i]
            # kp_vector = [kp.pt[0], kp.pt[1], kp.size, kp.angle]
            # kp_vector.extend(descriptors[i])
            # kp_matrix.append(kp_vector)
            kp_matrix.append(descriptors[i])

        return kp_matrix

    def extract_feature_feature_vectors(self):
        input_images = self.get_data_set_files()
        kp_feature_matrix = []
        kp_size = {}

        for img in input_images:
            kp_desc_matrix = self.extract_keypoints_descriptors(img.image_id)
            start_index = len(kp_feature_matrix)
            kp_size[img.image_id] = (start_index, start_index + len(kp_desc_matrix) - 1)

            for kp_desc in kp_desc_matrix:
                kp_feature_matrix.append(kp_desc)

        return kp_feature_matrix, kp_size

    def extract_feature_feature_vectors2(self):
        input_images = self.get_data_set_files()
        feature_matrix = []

        # extract from the existing features file
        lsh_index = lsh.LSHIndex()
        # hdf5_file = CURRENT_DIR + os.sep + '..' + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'feature_vectors_full_data.hdf5'
        pickle_file = CURRENT_DIR + os.sep + '..' + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'bovw.pkl'
        with open(pickle_file, 'rb') as fp:
            whole_matrix = pickle.load(fp)

        for i in input_images:
            img_idx = lsh_index.image_id_map[i.image_id]
            feature_vector = whole_matrix[img_idx]
            feature_matrix.append(feature_vector)

        return feature_matrix

    # def find_m_similar_images(self, reduced_data_matrix, reduced_query_img_vector, m, dimensionality_reduction_model):
    #     return self.find_m_similar_images_euc(reduced_data_matrix, reduced_query_img_vector, m)

    def find_m_similar_images(self, new_kp_matrix, new_kp_matrix_for_image, query_image_id, kp_indices, m):
        sorted_similar_images = []
        score_dict = {}
        for image in self._input_images_:
            number_of_good_match = 0
            if image.image_id == query_image_id:
                score_dict[query_image_id] = len(new_kp_matrix_for_image)
                continue

            for i in range(0, len(new_kp_matrix_for_image)):
                q_vector = new_kp_matrix_for_image[i]
                distances = []
                for k in range(kp_indices[image.image_id][0], kp_indices[image.image_id][1] + 1):
                    vector = new_kp_matrix[k]
                    d = self.euclidean_distance(q_vector, vector)
                    distances.append(d)

                distances = sorted(distances)
                if distances[0] > 0 and distances[1] / distances[0] >= 1.5:
                    number_of_good_match = number_of_good_match + 1

                if number_of_good_match > 0:
                    score_dict[image.image_id] = number_of_good_match

        # Sort and pick m most similar images
        # score_dict[query_image_id] = len(new_kp_matrix_for_image)
        if len(score_dict) > 0:
            counter = 0
            for key, value in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True):
                if counter < m:
                    image_match = TrainingImage(key, self.get_path_from_id(key))
                    result = ImageMatch(image_match, float(value))
                    sorted_similar_images.append(result)
                    # print('image: %s with score: %f' % (key, float(value)))
                else:
                    break
                counter = counter + 1
        else:
            print("No good matches has been found")

        return sorted_similar_images

    # Multi threaded matching
    # calculate similarity and return m best matching results
    # def find_m_similar_images(self, new_kp_matrix, new_kp_matrix_for_image, query_image_id, kp_indices, m):
    #     sorted_similar_images = []
    #     score_dict = {}
    #     score_dict[query_image_id] = len(new_kp_matrix_for_image)
    #
    #     number_of_threads = 10
    #
    #     # Splitting the query match tasks(multiple file match) into parts equal to number of threads
    #     array_chunk = np.array_split(self._input_images_, number_of_threads)
    #
    #     worker_threads = []
    #     for thread_id in range(number_of_threads):
    #         thread = threading.Thread(name='Thread #' + str(thread_id), target=self.compare_image_worker,
    #                                   args=(
    #                                       score_dict, query_image_id, new_kp_matrix, new_kp_matrix_for_image, kp_indices, array_chunk[thread_id]
    #                                   ), )
    #         worker_threads.append(thread)
    #         worker_threads[thread_id].start()
    #
    #     for thread in worker_threads:
    #         thread.join()
    #
    #     # Sort and pick m most similar images
    #     # score_dict[query_image_id] = len(new_kp_matrix_for_image)
    #     if len(score_dict) > 0:
    #         counter = 0
    #         for key, value in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True):
    #             if counter < m:
    #                 image_match = TrainingImage(key, self.get_path_from_id(key))
    #                 result = ImageMatch(image_match, float(value))
    #                 sorted_similar_images.append(result)
    #                 # print('image: %s with score: %f' % (key, float(value)))
    #             else:
    #                 break
    #             counter = counter + 1
    #     else:
    #         print("No good matches has been found")
    #
    #     return sorted_similar_images

    def compare_image_worker(self, score_dict, query_image_id, new_kp_matrix, new_kp_matrix_for_image, kp_indices, images_to_match):
        for image in self._input_images_:
            number_of_good_match = 0
            if image.image_id == query_image_id:
                continue

            for i in range(0, len(new_kp_matrix_for_image)):
                q_vector = new_kp_matrix_for_image[i]
                distances = []
                for k in range(kp_indices[image.image_id][0], kp_indices[image.image_id][1] + 1):
                    vector = new_kp_matrix[k]
                    d = self.euclidean_distance(q_vector, vector)
                    distances.append(d)

                distances = sorted(distances)
                if distances[0] > 0 and distances[1] / distances[0] >= 1.5:
                    number_of_good_match = number_of_good_match + 1

                if number_of_good_match > 0:
                    score_dict[image.image_id] = number_of_good_match

    # Experimental only - This model of feature vector did not perform well while computing similarity
    def extract_feature_vector(self, image_id, threshold):
        kp_threshold = threshold
        feature_vector = []

        image_file_path = self.get_path_from_id(image_id)
        img = cv.imread(image_file_path)
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(kp_threshold)
        key_points, descriptors = sift.detectAndCompute(grayscale_img, None)

        kp_count = len(key_points)
        if kp_count > kp_threshold:
            kp_count = kp_threshold

        # append all key points descriptors in one big vector
        for i in range(0, kp_count):
            for val in descriptors[i]:
                feature_vector.append(val)

        return feature_vector

    def find_kp_count(self, image_id):
        image_file_path = self.get_path_from_id(image_id)
        img = cv.imread(image_file_path)
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(grayscale_img, None)
        return len(kp)

    def extract_feature_vector_model2(self, image_id):
        kp_threshold = 30
        feature_vector = []

        image_file_path = self.get_path_from_id(image_id)
        img = cv.imread(image_file_path)
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(kp_threshold)
        key_points, descriptors = sift.detectAndCompute(grayscale_img, None)

        kp_count = len(key_points)
        if kp_count > kp_threshold:
            kp_count = kp_threshold

        for i in range(0, kp_count):
            kp = key_points[i]
            kp_vector = [kp.pt[0], kp.pt[1], kp.size, kp.angle]
            kp_vector.extend(descriptors[i])
            feature_vector.append(kp_vector)

        return feature_vector

    def extract_feature_feature_vectors_model2(self):
        input_images = self.get_data_set_files()
        feature_matrix = []

        kp_count = []
        for img in input_images:
            kp_count.append(self.find_kp_count(img.image_id))
        threshold = np.min(kp_count)

        for img in input_images:
            feature_vector = self.extract_feature_vector(img.image_id, threshold)
            feature_matrix.append(feature_vector)

        return feature_matrix, threshold
