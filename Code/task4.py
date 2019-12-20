from image_features.colormoments import ColorMoment
from image_features.HOG import HOG
from image_features.SIFT import SIFT
from image_features.FeatureModel import *
from utils.cli_options import *
from utils.distance_measures import *
from utils.similarity_measures import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import cv2 as cv
from math import ceil
import os
import pagerank as pagerank
import scipy.sparse as sparse
import pandas as pd
import csv
from image_features.LBP import LBP
import cvxopt
import cvxopt.solvers
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import task5 as lsh

CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..' + os.sep, 'Outputs', 'Task_4')


class ImageSimilarity:
    def __init__(self, image_id, similarity):
        self.image_id = image_id
        self.similarity = similarity


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth

    def _best_split(self, X, y):
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth = 0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini = self._gini(y),
            num_samples = y.size,
            num_samples_per_class = num_samples_per_class,
            predicted_class = predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))


def dimension_reduction(X, k, selected_model, dim_reduction_tech, input_images):
    model = get_feature_model(selected_model)
    dim_reduction_model = get_dim_reduction_tech(dim_reduction_tech)

    # HOG (ideal = 73%, edges = 12, dim = 50, data set 2->2)
    # top_dim = 50
    # CM (ideal = 72%, edges = 12, dim = 20, data set 2->2)
    # data set 2 -> 1 = 80%
    # data set 1 -> 2 = 48%
    top_dim = 20

    # PCA
    if dim_reduction_tech == "1":
        pca = PCA(n_components = top_dim)
        U, _, V = pca._fit(X)
        latent_features = U[:, :top_dim]

    # SVD
    elif dim_reduction_tech == "2":
        # U, _, V = np.linalg.svd(X)
        svd = TruncatedSVD(n_components=top_dim)
        U = svd.fit_transform(np.array(X))
        latent_features = U[:, :top_dim]

    # STORE THE SIMILARITY DICT IN H5py or pickle object beforehand
    # Find the similarity
    similarity_dict = {}
    lf_len = len(latent_features)
    # input_images += training_input_images

    for i in range(lf_len):
        for j in range(lf_len):
            if i == j:
                continue

            dist = euclidean_distance(list(latent_features[i]), list(latent_features[j]))
            sim = round((1 / (dist+1)), 4)
            # sim = cosine_similarity(list(latent_features[i]), list(latent_features[j]))
            img_sim = ImageSimilarity(input_images[j].image_id, sim)

            if input_images[i].image_id not in similarity_dict:
                similarity_dict[input_images[i].image_id] = [img_sim]
            else:
                similarity_dict[input_images[i].image_id].append(img_sim)

    # Sort the similarity graph (dictionary)
    for key, val in similarity_dict.items():
        similarity_dict[key] = sorted(val, key = lambda x: x.similarity, reverse = True)
        # chose the top k related images
        similarity_dict[key] = similarity_dict[key][:k]

    # Normalize the values to sum of 1


    # print the k similar images
    # for k,v in similarity_dict.items():
    #     print('Image ID: ', k)
    #     for i in v:
    #         print(i.image_id, i.similarity)
    #     print('\n')

    return similarity_dict


def dimension_reduction_sift(X, k, selected_model, dim_reduction_tech, input_images):
    model = get_feature_model(selected_model)
    dim_reduction_model = get_dim_reduction_tech(dim_reduction_tech)

    top_dim = 50

    # PCA
    # if dim_reduction_tech == "1":
    #     pca = PCA(n_components = top_dim)
    #     X = np.array(X)
    #     U = pca.fit_transform(X)
    #     latent_features = U[:, :top_dim]
    #
    # # SVD
    # elif dim_reduction_tech == "2":
    #     # U, _, V = np.linalg.svd(X)
    #     svd = TruncatedSVD(n_components=top_dim)
    #     U = svd.fit_transform(np.array(X))
    #     latent_features = U[:, :top_dim]

    # STORE THE SIMILARITY DICT IN H5py or pickle object beforehand
    # Find the similarity
    similarity_dict = {}
    # lf_len = len(latent_features)
    # input_images += training_input_images

    bf = cv.BFMatcher()
    for i in range(len(input_images)):
        des1 = X[i]
        for j in range(len(input_images)):
            if input_images[i].image_id == input_images[j].image_id:
                continue

            des2 = X[j]
            matches = bf.knnMatch(des1, des2, k=2)
            good_count = 0
            for m, n in matches:
                # Apply ratio test
                if m.distance < 0.85 * n.distance:
                    # good.append([m])
                    good_count += 1

            img_sim = ImageSimilarity(input_images[j].image_id, good_count)

            if input_images[i].image_id not in similarity_dict:
                similarity_dict[input_images[i].image_id] = [img_sim]
            else:
                similarity_dict[input_images[i].image_id].append(img_sim)

    # Sort the similarity graph (dictionary)
    for key, val in similarity_dict.items():
        similarity_dict[key] = sorted(val, key=lambda x: x.similarity, reverse=True)
        # similarity_dict[key] = sorted(val, key = lambda x: x.similarity)
        # chose the top k related images
        similarity_dict[key] = similarity_dict[key][:k]

    return similarity_dict


# Dimensionality reduction for Decision Tree Classifier
def reduce_dimensions_dt(X, k, dim_red_tech):

    if dim_red_tech == 1:
        pca = PCA(n_components=k)
        X_reduced = pca.fit_transform(X)
        return X_reduced
    elif dim_red_tech == 2:
        svd = TruncatedSVD(n_components=k)
        X_reduced = svd.fit_transform(np.array(X))
        return X_reduced


def ppr(similarity_dict, images, K):
    related_images = []
    weights = []
    personalize = []
    image_id_map = {}
    temp = 0
    for k, v in similarity_dict.items():
        image_id_map[k] = temp
        temp += 1

    for k, v in similarity_dict.items():
        found = False
        for j in images:
            if j.image_id == k:
                personalize.append(1/(len(images)))
                found = True
                break
        if not found :
           personalize.append(0)

        for i in v:
            edge = []
            edge.append(image_id_map[k])
            edge.append(image_id_map[i.image_id])
            related_images.append(edge)
            weights.append(i.similarity)
        # print('\n')
    A = np.array(related_images)
    weights = np.array(weights)
    G = sparse.csr_matrix((weights, (A[:, 0], A[:, 1])), shape=(len(similarity_dict), len(similarity_dict)))
    personalize = np.array(personalize)
    # pr = pagerank.pagerank_power(G, p=0.85, personalize=personalize, tol=1e-6, max_iter=200)
    pr = pagerank.personalizedPageRank(G, personalize, c=0.15, allowedDiff=1e-9, maxIters=200)
    pr_sorted = np.argsort(pr)
    pr_sorted = pr_sorted[::-1]

    img_ppr = []
    # print('\nPersonalized Page ranking are:\n-------------------------------------------')

    for t in range(0, K):
        id = pr_sorted[t]
        score = pr[id]
        for image_id, node_id in image_id_map.items():
            if node_id == id:
                # no need to print score for the image
                # print('\n')
                # print('Image ID ' + str(t + 1) + ': ' + image_id + ' : ' +  str(score))

                # Store the information to display it using matplot
                temp = [image_id, score]
                img_ppr.append(temp)
    return img_ppr


def polynomial_kernel(x1,x2,p =3):
    return (1 + np.dot(x1, x2)) ** p


def rbf_kernel(x1,x2):
    return np.exp(-0.5*np.sum((x1-x2)**2))


def run_task():
    np.random.seed(81)

    # TODO: changing this will impact other classifiers so should be used with respect to classifier
    # selected_model = choose_feature_model()
    selected_model = "1"
    # dim_reduction_tech = choose_dim_reduction_tech()
    dim_reduction_tech = "1"
    classifier_type = choose_classifier()
    
    folder_path, dataset_type = input_data_set_folder_path(task=4)
    training_folder_path = input_data_set_folder_path(type='training')

    if dataset_type == 1:
        labelled_file = os.path.join(CURRENT_DIR, '..' + os.sep, 'Inputs' + os.sep + 'labelled_set1.csv')
    elif dataset_type == 2:
        labelled_file = os.path.join(CURRENT_DIR, '..' + os.sep, 'Inputs' + os.sep + 'labelled_set2.csv')
    elif dataset_type == 3:
        labelled_file = os.path.join(CURRENT_DIR, '..' + os.sep, 'Inputs' + os.sep + 'unlabelled_set1.csv')
    elif dataset_type == 4:
        labelled_file = os.path.join(CURRENT_DIR, '..' + os.sep, 'Inputs' + os.sep + 'unlabelled_set2.csv')

    print("\nTraining the classifier...")

    # TODO: add SVM classifier code
    if classifier_type == "1":
        dorsal_images, palmar_images = get_dorsal_palmar_images()
        test_image_feature_model = SIFT(training_folder_path)
        test_input_images = test_image_feature_model.get_data_set_files()
        test_feature_matrix = test_image_feature_model.extract_feature_feature_vectors2()

        label_image_feature_model = SIFT(folder_path)
        label_input_images = label_image_feature_model.get_data_set_files()
        feature_matrix = label_image_feature_model.extract_feature_feature_vectors2()
        model = get_feature_model(selected_model)
        top_dim = 50
        svd = TruncatedSVD(n_components=top_dim)
        svd.fit(np.array(feature_matrix))
        U = svd.transform(np.array(feature_matrix))
        latent_features = U[:, :top_dim]
        test_feature_matrix_reduced = svd.transform(np.array(test_feature_matrix))
        latent_features_unlabelled = test_feature_matrix_reduced[:,:top_dim]
        labels = []
        meta_deta = pd.read_csv(os.path.join(CURRENT_DIR, '..' + os.sep, 'Metadata', 'HandInfo.csv')) #This needs to change to take the input from the labelled set in the input folder

        test_labels = []
        for i in range(len(latent_features)):
            img_id = label_input_images[i].image_id
            aspect = meta_deta[meta_deta['imageName']=='Hand_'+img_id+'.jpg']['aspectOfHand'].iloc[0]
            if 'palmar' in aspect:          #Palmar is 1 and Dorsal is -1
                labels.append(1)
            else:
                labels.append(-1)

        for i in range(len(test_input_images)):
            img_id = test_input_images[i].image_id
            aspect = meta_deta[meta_deta['imageName']=='Hand_'+img_id+'.jpg']['aspectOfHand'].iloc[0]
            if 'palmar' in aspect:          #Palmar is 1 and Dorsal is -1
                test_labels.append(1)
            else:
                test_labels.append(-1)
        
        labels = np.array(labels)
        C = 10
        X = latent_features
        n = X.shape[0] #len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i][j] = labels[i] * labels[j] * polynomial_kernel(X[i], X[j])
        P = cvxopt_matrix(K)
        q = cvxopt_matrix(np.ones(n) * -1)
        A = labels.reshape((1, n))
        A = cvxopt_matrix(A.astype('double'))
        # A = matrix(y, (1, n), tc='d')
        b = cvxopt_matrix(0, tc='d')
        temp1 = np.identity(n) * -1
        temp2 = np.identity(n)
        G = cvxopt_matrix(np.vstack((temp1, temp2)))
        temp3 = cvxopt_matrix(np.zeros(n))
        temp4 = cvxopt_matrix(np.ones(n) * C)
        h = cvxopt_matrix(np.vstack((temp3, temp4)))
        cvxopt_solvers.options['show_progress'] = False
        solution = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).reshape(n)
        support_vectors = np.where(alphas > 1e-7)[0][0]
        b = labels[support_vectors] - sum(alphas * labels * polynomial_kernel(X, X[support_vectors]))
        predict = np.zeros(len(latent_features_unlabelled))

        for i in range(len(latent_features_unlabelled)):
            asum = 0
            vector = latent_features_unlabelled[i]
            asum = sum(alphas*labels*polynomial_kernel(latent_features,vector)) + b
            predict[i] = np.sign(asum)
        results = predict
        
        tp = 0

        print("\nLabeling unlabelled images...")
        print("\nImage ID : Label")
        for i in range(len(results)):
            # if i <= 50 and results[i] == -1 :
            if results[i] == test_labels[i] :
                tp +=1
            # elif i > 50 and results[i] == 1 :
            #     tp +=1
            img_id = test_input_images[i].image_id
            if results[i] == -1:
                print(img_id+ '  : Dorsal')
            elif results[i] == 1:
                print(img_id + '  : Palmar')
        
        print('\nClassifier Accuracy: ' + str((tp / 100) * 100))

        #################################### DECISION TREE ##############################################
    elif classifier_type == "2":
        labelled_set = folder_path[-1]
        unlabelled_set = training_folder_path[-1]

        # INPUT FOLDERS & METADATAS
        labelled_folder_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                            'Inputs', 'Labelled', 'Set' + labelled_set)
        labelled_metadata_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                              'Inputs', 'labelled_set' + labelled_set + '.csv')
        labelled_metadata = pd.read_csv(labelled_metadata_path, index_col = "id")

        unlabelled_folder_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                              'Inputs', 'Unlabelled', 'Set ' + unlabelled_set)

        # LABELLED

        print('Training Decision Tree Classifier ...')

        labelled_feature_model = FeatureModel(labelled_folder_path)
        labelled_images = labelled_feature_model.get_data_set_files()

        target = []

        for image in labelled_images:
            aspect = \
                labelled_metadata[labelled_metadata['imageName'] == 'Hand_' + image.image_id + '.jpg'][
                    'aspectOfHand'].iloc[0]
            if 'palmar' in aspect:
                target.append(1)
            if 'dorsal' in aspect:
                target.append(0)

        labelled_feature_model = ColorMoment(labelled_folder_path)

        feature_matrix = labelled_feature_model.extract_feature_feature_vectors_list(labelled_images)

        X = np.array(feature_matrix)
        y = np.array(target)
        X_reduced = reduce_dimensions_dt(X, 20, 2)

        clf = DecisionTreeClassifier(max_depth = 5)
        clf.fit(X_reduced, y)

        print('Training Decision Tree Classifier ... DONE.')

        # UNLABELLED

        print('Classifying unlabelled images...')

        unlabelled_feature_model = FeatureModel(unlabelled_folder_path)
        unlabelled_images = unlabelled_feature_model.get_data_set_files()

        unlabelled_feature_model = ColorMoment(unlabelled_folder_path)

        unlabelled_feature_matrix = unlabelled_feature_model.extract_feature_feature_vectors_list(unlabelled_images)
        X_unlabelled = np.array(unlabelled_feature_matrix)
        X_unlabelled_reduced = reduce_dimensions_dt(X_unlabelled, 20, 2)

        predictions = clf.predict(X_unlabelled_reduced)
        aspect_dict = {}

        # Used to calculate accuracy of the classification
        dorsal_images, palmar_images = get_dorsal_palmar_images()
        hits = 0

        for image, prediction in zip(unlabelled_images, predictions):

            if prediction == 0:
                aspect_dict[image.image_id] = 'dorsal'
                if image.image_id in dorsal_images:
                    hits += 1
            else:
                aspect_dict[image.image_id] = 'palmar'
                if image.image_id in palmar_images:
                    hits += 1

            print(f'{image.image_id} : {aspect_dict[image.image_id]}')

        print('Classifying unlabelled images... DONE.')

        accuracy = (hits / len(unlabelled_images)) * 100
        print(f'\nAccuracy = {accuracy}%')

        OUTPUT_PATH = os.path.join(OUTPUT_DIR,
                                   'DT_labelledset' + labelled_set +
                                   '_unlabelledset' + unlabelled_set +
                                   '.csv')

        with open(OUTPUT_PATH, 'w', newline = "") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in aspect_dict.items():
                writer.writerow([key, value])

            print('(ImageID,AspectOfHand) info saved in file - ', OUTPUT_PATH)

    # PPR based classifier code
    elif classifier_type == "3":
        # k = int(input('Enter the value of k (outgoing edges): '))
        # for cm + pca
        # k = 12
        k = 18
        lsh_index = lsh.LSHIndex(load_sift=True)

        if selected_model == "1":
            unlabelled_image_feature_model = ColorMoment(training_folder_path)
            unlabelled_input_images = unlabelled_image_feature_model.get_data_set_files()
            unlabelled_feature_matrix = unlabelled_image_feature_model.extract_feature_feature_vectors()

            # Dorsal
            dor_image_feature_model = ColorMoment(folder_path, image_label=ImageLabel.DORSAL,
                                                  labelled_file=labelled_file)
            dor_input_images = dor_image_feature_model.get_data_set_files()
            dor_feature_matrix = dor_image_feature_model.extract_feature_feature_vectors()
            # we have combined features matrix with dorsal, palmar, and ulabelled, so no longer used
            X1 = np.array(dor_feature_matrix + unlabelled_feature_matrix)
            # similarity_dict_dor = dimension_reduction(X1, k, selected_model, dim_reduction_tech,
            #                                          dor_input_images, training_input_images)

            # Palmar
            palm_image_feature_model = ColorMoment(folder_path, image_label=ImageLabel.PALMAR,
                                                   labelled_file=labelled_file)
            palm_input_images = palm_image_feature_model.get_data_set_files()
            palm_feature_matrix = palm_image_feature_model.extract_feature_feature_vectors()
            # we have combined features matrix with dorsal, palmar, and ulabelled, so no longer used
            X2 = np.array(palm_feature_matrix + unlabelled_feature_matrix)
            # similarity_dict_palm = dimension_reduction(X2, k, selected_model, dim_reduction_tech,
            #                                            palm_input_images,
            #                                            training_input_images)

            X = np.array(dor_feature_matrix + palm_feature_matrix + unlabelled_feature_matrix)
            all_images = dor_input_images + palm_input_images + unlabelled_input_images
            # similarity_dict = dimension_reduction(X, k, selected_model, dim_reduction_tech, all_images)

            # Using SIFT to find similarity graph before creating the PPR graph
            all_image_features = []
            for image in all_images:
                img_id = image.image_id
                img_feature = []
                img_index = lsh_index.image_id_map[img_id]
                start_point = lsh_index.data_sift_kp_starts[img_index]
                if img_index > len(lsh_index.data_sift_kp_starts):
                    img_feature = lsh_index.data_sift[start_point:]
                    all_image_features.append(img_feature)
                else:
                    end_point = lsh_index.data_sift_kp_starts[img_index + 1]
                    img_feature = lsh_index.data_sift[start_point:end_point]
                all_image_features.append(img_feature)
            similarity_dict = dimension_reduction_sift(all_image_features, k, selected_model, dim_reduction_tech,
                                                       all_images)
            # SIFT ends

            # PPR
            # print(len(X1))
            dor_img_ppr = ppr(similarity_dict, dor_input_images, len(X))
            palm_img_ppr = ppr(similarity_dict, palm_input_images, len(X))

            print("\nLabeling unlabelled images...")
            print("\nImage ID : Label")
            dor_score, palm_score = 0, 0
            tp = 0
            # Used to calculate accuracy of the classification
            dorsal_images, palmar_images = get_dorsal_palmar_images()

            for i in unlabelled_input_images:
                for j in range(len(dor_img_ppr)):
                    if dor_img_ppr[j][0] == i.image_id:
                        # dor_score = dor_img_ppr[j][1]
                        dor_score = j
                        break

                for j in range(len(palm_img_ppr)):
                    if palm_img_ppr[j][0] == i.image_id:
                        # palm_score = palm_img_ppr[j][1]
                        palm_score = j
                        break

                # if dor_score > palm_score:
                if dor_score < palm_score:
                    print(i.image_id + '  : Dorsal')
                    # check for true positive
                    if i.image_id in dorsal_images:
                        tp += 1
                else:
                    print(i.image_id + '  : Palmar')
                    # check for true positive
                    if i.image_id in palmar_images:
                        tp += 1

            print('\nClassifier Accuracy: ' + str((tp / 100) * 100))

        elif selected_model == "3":
            unlabelled_image_feature_model = HOG(training_folder_path)
            unlabelled_input_images = unlabelled_image_feature_model.get_data_set_files()
            unlabelled_feature_matrix = unlabelled_image_feature_model.extract_feature_feature_vectors()

            # Dorsal
            dor_image_feature_model = HOG(folder_path, image_label=ImageLabel.DORSAL,
                                                  labelled_file=labelled_file)
            dor_input_images = dor_image_feature_model.get_data_set_files()
            dor_feature_matrix = dor_image_feature_model.extract_feature_feature_vectors()
            # we have combined features matrix with dorsal, palmar, and ulabelled, so no longer used
            X1 = np.array(dor_feature_matrix + unlabelled_feature_matrix)
            # similarity_dict_dor = dimension_reduction(X1, k, selected_model, dim_reduction_tech,
            #                                          dor_input_images, training_input_images)

            # Palmar
            palm_image_feature_model = HOG(folder_path, image_label=ImageLabel.PALMAR,
                                                   labelled_file=labelled_file)
            palm_input_images = palm_image_feature_model.get_data_set_files()
            palm_feature_matrix = palm_image_feature_model.extract_feature_feature_vectors()
            # we have combined features matrix with dorsal, palmar, and ulabelled, so no longer used
            X2 = np.array(palm_feature_matrix + unlabelled_feature_matrix)
            # similarity_dict_palm = dimension_reduction(X2, k, selected_model, dim_reduction_tech,
            #                                            palm_input_images,
            #                                            training_input_images)

            X = np.array(dor_feature_matrix + palm_feature_matrix + unlabelled_feature_matrix)
            all_images = dor_input_images + palm_input_images + unlabelled_input_images
            similarity_dict = dimension_reduction(X, k, selected_model, dim_reduction_tech, all_images)

            # PPR
            # print(len(X1))
            dor_img_ppr = ppr(similarity_dict, dor_input_images, len(X))
            palm_img_ppr = ppr(similarity_dict, palm_input_images, len(X))

            print("\nLabeling unlabelled images...")
            print("\nImage ID : Label")
            dor_score, palm_score = 0, 0
            tp = 0
            # Used to calculate accuracy of the classification
            dorsal_images, palmar_images = get_dorsal_palmar_images()

            for i in unlabelled_input_images:
                for j in range(len(dor_img_ppr)):
                    if dor_img_ppr[j][0] == i.image_id:
                        # dor_score = dor_img_ppr[j][1]
                        dor_score = j
                        break

                for j in range(len(palm_img_ppr)):
                    if palm_img_ppr[j][0] == i.image_id:
                        # palm_score = palm_img_ppr[j][1]
                        palm_score = j
                        break

                # if dor_score > palm_score:
                if dor_score < palm_score:
                    print(i.image_id + '  : Dorsal')
                    # check for true positive
                    if i.image_id in dorsal_images:
                        tp += 1
                else:
                    print(i.image_id + '  : Palmar')
                    # check for true positive
                    if i.image_id in palmar_images:
                        tp += 1

            print('\nClassifier Accuracy: ' + str((tp / 100) * 100))


if __name__ == "__main__":
    run_task()