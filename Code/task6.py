from utils.cli_options import *
from utils.similarity_measures import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import os
import pickle
import pagerank as pagerank
import scipy.sparse as sparse
import h5py
import task5 as lsh
import cvxopt
import cvxopt.solvers
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


CURRENT_DIR = os.path.dirname(__file__)
IMG_PREFIX = '/Hand_'
IMG_EXT = '.jpg'
OUTPUT_DIR = CURRENT_DIR + os.sep + '..' + os.sep + 'Outputs'


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


def input_rel_irrelevant_image_ids(is_relevant=True, relevant_set=None):
    image_ids = []
    if is_relevant:
        print('Enter relevant image ids as comma separated value:')
    else:
        print('Enter irrelevant image ids as comma separated value:')
    input_str = input()
    input_ids = []
    if input_str.strip() != '':
        input_ids = [x.strip() for x in input_str.split(',')]

    if is_relevant:
        if not relevant_set:
            relevant_set = []
        for img_id in input_ids:
            if img_id not in relevant_set:
                image_ids.append(img_id)
    else:
        if not relevant_set:
            relevant_set = []

        for img_id in input_ids:
            if img_id in relevant_set:
                relevant_set.remove(img_id)
            image_ids.append(img_id)

    return image_ids, relevant_set

    # so that we can avoid duplicates
    # if image_id not in image_ids:
    #     image_ids.append(image_id)
    #
    # while image_id != '-1':
    #     image_id = input()
    #     if image_id == '-1':
    #         break
    #     image_ids.append(image_id)
    #
    # return image_ids


class ImageSimilarity:
    def __init__(self, image_id, similarity):
        self.image_id = image_id
        self.similarity = similarity


def dimension_reduction(X, k, selected_model, dim_reduction_tech, input_images):
    model = get_feature_model(selected_model)
    dim_reduction_model = get_dim_reduction_tech(dim_reduction_tech)

    top_dim = 10

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

    # TODO: STORE THE SIMILARITY DICT IN H5py or pickle object beforehand
    # Find the similarity
    similarity_dict = {}
    lf_len = len(latent_features)

    for i in range(lf_len):
        for j in range(lf_len):
            if i == j:
                continue

            # dist = euclidean_distance(list(latent_features[i]), list(latent_features[j]))
            sim = cosine_similarity(list(latent_features[i]), list(latent_features[j]))
            # sim = round((1 / dist) * 1000, 4)
            # Here we just have the input image ids not the input image object as we don't depend on the data set
            img_sim = ImageSimilarity(input_images[j], sim)

            if input_images[i] not in similarity_dict:
                similarity_dict[input_images[i]] = [img_sim]
            else:
                similarity_dict[input_images[i]].append(img_sim)

    # Sort the similarity graph (dictionary)
    for key, val in similarity_dict.items():
        similarity_dict[key] = sorted(val, key = lambda x: x.similarity, reverse = True)
        # chose the top k related images
        similarity_dict[key] = similarity_dict[key][:k]

    # Normalize the values to sum of 1

    return similarity_dict


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
            if j == k:
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
    # pr = pagerank.pagerank_power(G, p=0.85, personalize=None, tol=1e-6, max_iter=200)
    pr = pagerank.personalizedPageRank(G, personalize, c=0.20, allowedDiff=1e-9, maxIters=200)
    pr_sorted = np.argsort(pr)
    pr_sorted = pr_sorted[::-1]

    img_ppr = []
    # print('\nPersonalized Page ranking are:\n-------------------------------------------')

    # for t in range(0, K):
    for t in range(len(pr_sorted)):
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


# print formatted search results
def print_ranked_results(similarity_list, relevant_images=None):
    print('\nRanked result set from task 5:')
    print('Image ID:   Score')

    if relevant_images:
        rel_img_id = relevant_images[0]
        lsh_index = lsh.LSHIndex()
        q_img_idx = lsh_index.image_id_map[rel_img_id]

        for i in range(len(similarity_list)):
            print(similarity_list[i][0] + ' : ' + str(similarity_list[i][2]))
            # if similarity_list[i][0] == rel_img_id:
            #     q_img_idx = i

        satisfied = input("\nDo you want to visualize the result set? [Enter 'yes' or 'no']\n")
        if satisfied == 'yes':

            lsh_index.visualize_similar_results(q_img_idx, len(similarity_list), similarity_list)
    else:
        for i in range(len(similarity_list)):
            print(similarity_list[i][0] + ' : ' + str(similarity_list[i][2]))

    print()

# print formatted search results
def print_ranked_results_v2(query_image_id, ranked_image_ids, scores, total_images):
    print('\nRanked result set from task 5:')
    print('Image ID:   Score')

    lsh_index = lsh.LSHIndex()

    counter = 0
    for ranked_image_id, score in zip(ranked_image_ids, scores):
        if counter == total_images:
            break
        print(ranked_image_id + ' : ' + str(score))
        counter += 1

    is_visualize = input("\nDo you want to visualize the result set? [Enter 'yes' or 'no']\n")
    if is_visualize == 'yes':
        lsh_index.visualize_similar_results_v2(query_image_id, ranked_image_ids, scores, total_images)

    print()

def polynomial_kernel(x1,x2,p =3):
    return (1 + np.dot(x1, x2)) ** p


def rbf_kernel(x1,x2):
    return np.exp(-0.5*np.sum((x1-x2)**2))

def run_task():
    # selected_model = choose_feature_model()
    selected_model = "1"
    # dim_reduction_tech = choose_dim_reduction_tech()
    dim_reduction_tech = "2"
    rel_feed_model = choose_relevance_feedback_model()

    # read task 5 output - ranked images
    similarity_list = None
    similarity_dict = {}
    T = -1
    ranked_res_pickle = OUTPUT_DIR + os.sep + 'Task_5' + os.sep + 'task5_output.pkl'
    if os.path.exists(ranked_res_pickle):
        with open(ranked_res_pickle, 'rb') as handle:
            similarity_list = pickle.load(handle)
    T = len(similarity_list)

    # lsh_similarity_list = list(similarity_list)

    # has feature vectors for similar images only
    feature_matrix = []
    # display existing ranked results for the user to pick relevant and irrelevant images
    # console_msg = 'Image ID:  Score    Image ID:  Score    Image ID:  Score    Image ID:  Score    Image ID:  Score    ' \
    #               'Image ID:  Score    Image ID:  Score    Image ID:  Score'

    hdf5_file = CURRENT_DIR + os.sep + '..' + os.sep + 'Metadata' + os.sep + 'feature_vectors_full_data.hdf5'
    data_cm = None
    with h5py.File(hdf5_file, 'r') as hf:
        data_cm = hf['hog_features'][:]

    # method call to format and print search results
    # passing first result as relevant set so that it can be used for html visualization
    print_ranked_results(similarity_list, [similarity_list[0][0]])

    ranked_image_ids = []
    for i in range(len(similarity_list)):
        # if i % 8 == 0:
        #     console_msg += '\n'
        # else:
        #     console_msg += ',  '
        # console_msg += similarity_list[i][0] + (" :  %.7f" % (similarity_list[i][2]))

        similarity_dict[similarity_list[i][0]] = similarity_list[i][2]

        h5_image_idx = similarity_list[i][1]
        feature_vector = data_cm[h5_image_idx]
        feature_matrix.append(feature_vector)

        ranked_image_ids.append(similarity_list[i][0])

    # print('Ranked result set from task 5:')
    # print(console_msg + '\n')

    # get user input for relevant and irrelevant images
    # the relevant image set will always have the initial user query image id
    relevant_set = [ranked_image_ids[0]]
    relevant_set_temp, _ = input_rel_irrelevant_image_ids()  # is a list of image_id's
    relevant_set += relevant_set_temp
    irrelevant_set, relevant_set = input_rel_irrelevant_image_ids(is_relevant=False, relevant_set=relevant_set)

    lsh_index = lsh.LSHIndex()
    data_hog_ = lsh_index.data_hog_
    binary_data_hog_ = np.ones(data_hog_.shape) * (data_hog_ > np.mean(data_hog_, axis=0))

    # TODO: add SVM relevance feedback system code
    if rel_feed_model == "1":
        provide_rel_feedback = 'yes'
        top_irr_to_discard = 5
        t_rel = T

        query_image_id = relevant_set[0]
        query_image_features = feature_matrix[0]

        while 'yes' in provide_rel_feedback:

            #top_results_considered = T // len(relevant_set)
            X = []
            y = []
            # Relevant = 0
            # Irrelevant = 1
            for ranked_image_id, feature_row in zip(ranked_image_ids, feature_matrix):
                if ranked_image_id in relevant_set:
                    y.append(1)
                    X.append(feature_row)
                elif ranked_image_id in irrelevant_set:
                    y.append(-1)
                    X.append(feature_row)

            X = np.array(X)
            y = np.array(y)
            n = X.shape[0] #len(X)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i][j] = y[i] * y[j] * polynomial_kernel(X[i], X[j])
            P = cvxopt_matrix(K)
            q = cvxopt_matrix(np.ones(n) * -1)
            A = y.reshape((1, n))
            A = cvxopt_matrix(A.astype('double'))
            C = 5
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
            b = y[support_vectors] - sum(alphas * y * polynomial_kernel(X, X[support_vectors]))
            # Model fit
            
            
            irrelevant_related_images = []
            for i in range(len(irrelevant_set)):
                results_for_this_image = lsh_index.query_image(irrelevant_set[i])
                # print(results_for_this_image, irrelevant_set[i],' are the results')
                if len(results_for_this_image) > top_irr_to_discard:
                    results_for_this_image = results_for_this_image[:top_irr_to_discard]

                for result in results_for_this_image:
                    irrelevant_related_images.append(result[0])

            # TODO: make sure to have query image vector as well
            # best we can include all the results from
            # do lsh using irrelevant as well and remove some images from actual image query
            # and remove unwanted from actual result set
            rel_result_sets = {}
            images_included_in_result_set = []
            xyz = 0
            for i in range(len(relevant_set)):
                results_for_this_image = lsh_index.query_image(relevant_set[i])
                result_count_included = 0
                top_results = []
                #for result in results_for_this_image[:top_results_considered]:
                for result in results_for_this_image:
                    # if result_count_included > t_rel:
                    #     break

                    if result[0] not in images_included_in_result_set:
                        top_results.append(result)
                        images_included_in_result_set.append(result[0])
                        result_count_included += 1

                # if len(results_for_this_image) > t_rel:
                #     results_for_this_image = results_for_this_image[:t_rel]

                # rel_result_sets[relevant_set[i]] = results_for_this_image
                rel_result_sets[relevant_set[i]] = top_results
                xyz += len(top_results)

            for rel_id,image_list in rel_result_sets.items():
                temp = []
                for im in image_list:
                    if im not in irrelevant_related_images:
                        temp.append(im)

                rel_result_sets[rel_id] = temp


            feature_matrix = []
            new_image_ids = []
            for k, v in rel_result_sets.items():
                for result in v:
                    h5_image_idx = result[1]
                    img_id = result[0]
                    if img_id not in irrelevant_set:
                        new_image_ids.append(img_id)
                        feature_vector = data_cm[h5_image_idx]
                        feature_matrix.append(feature_vector)


            latent_features_unlabelled = feature_matrix
            predict = np.zeros(len(latent_features_unlabelled))

            for i in range(len(latent_features_unlabelled)):
                asum = 0
                vector = latent_features_unlabelled[i]
                asum = sum(alphas*y*polynomial_kernel(X,vector)) + b
                predict[i] = np.sign(asum)
            results = predict
            #Prediction done

            final_image_ids = []
            final_feature_matrix = []

            for new_image_id, feature_vector, prediction in zip(new_image_ids, feature_matrix, results):
                # relevant
                if prediction == 1:
                    final_image_ids.append(new_image_id)
                    final_feature_matrix.append(feature_vector)

            # distance of each new image ID(all relevant) from query image
            final_image_distances = [np.linalg.norm(feature_vector - query_image_features)
                                     for feature_vector in final_feature_matrix]

            # sorting relevant image IDs based on distance from query image
            ranked_image_ids = [x for _, x in sorted(zip(final_image_distances, final_image_ids), key=lambda pair: pair[0])]
            feature_matrix = [x for _, x in sorted(zip(final_image_distances, final_feature_matrix), key=lambda pair: pair[0])]
            final_image_distances.sort()

            final_image_scores = []
            for final_image_distance in final_image_distances:
                final_image_scores.append(1 / (1+final_image_distance))


            print_ranked_results_v2(query_image_id, ranked_image_ids, final_image_scores, t_rel)
            # print('New results based on relevace feedback(renewd query formation):\nImage ID : Score')
            # for ranked_image_id, final_image_score in zip(ranked_image_ids, final_image_scores):
            #     print(f'{ranked_image_id} : {final_image_score}')

            provide_rel_feedback = input("Do you want to provide relevance feedback? [Enter 'yes' or 'no']\n")

            if 'yes' in provide_rel_feedback:
                # get user input for relevant and irrelevant images
                # relevant_set += input_rel_irrelevant_image_ids()
                # irrelevant_set += input_rel_irrelevant_image_ids(is_relevant = False)
                relevant_set_temp, _ = input_rel_irrelevant_image_ids()  # is a list of image_id's
                relevant_set += relevant_set_temp
                irrelevant_set_temp, relevant_set = input_rel_irrelevant_image_ids(is_relevant=False,
                                                                              relevant_set=relevant_set)
                irrelevant_set += irrelevant_set_temp

            
    #################################### DECISION TREE ####################################
    elif rel_feed_model == "2":

        provide_rel_feedback = 'yes'
        t_rel = T #len(similarity_list)
        top_irr_to_discard = 5

        query_image_id = relevant_set[0]
        query_image_features = feature_matrix[0]

        while 'yes' in provide_rel_feedback:

            top_results_considered = T // len(relevant_set)

            X = []
            y = []
            # Relevant = 0
            # Irrelevant = 1
            for ranked_image_id, feature_row in zip(ranked_image_ids, feature_matrix):
                if ranked_image_id in relevant_set:
                    y.append(0)
                    X.append(feature_row)
                elif ranked_image_id in irrelevant_set:
                    y.append(1)
                    X.append(feature_row)

            X = np.array(X)
            y = np.array(y)

            clf = DecisionTreeClassifier(max_depth = 5)
            clf.fit(X, y)

            # NEW LOGIC

            irrelevant_related_images = []
            for i in range(len(irrelevant_set)):
                results_for_this_image = lsh_index.query_image(irrelevant_set[i])
                if len(results_for_this_image) > top_irr_to_discard:
                    results_for_this_image = results_for_this_image[:top_irr_to_discard]

                for result in results_for_this_image:
                    irrelevant_related_images.append(result[0])

            # # TODO: make sure to have query image vector as well
            # # best we can include all the results from
            # # do lsh using irrelevant as well and remove some images from actual image query
            # # and remove unwanted from actual result set
            # rel_result_sets = {}
            # images_included_in_result_set = []
            # for i in range(len(relevant_set)):
            #     results_for_this_image = lsh_index.query_image(relevant_set[i])
            #     result_count_included = 0
            #     top_results = []
            #     for result in results_for_this_image[:top_results_considered]:
            #         if result_count_included > t_rel:
            #             break
            #         if result[0] not in images_included_in_result_set:
            #             top_results.append(result)
            #             images_included_in_result_set.append(result[0])
            #             result_count_included += 1
            #
            #     rel_result_sets[relevant_set[i]] = top_results

            # TODO: make sure to have query image vector as well
            # best we can include all the results from
            # do lsh using irrelevant as well and remove some images from actual image query
            # and remove unwanted from actual result set
            rel_result_sets = {}
            images_included_in_result_set = []
            xyz = 0
            for i in range(len(relevant_set)):
                results_for_this_image = lsh_index.query_image(relevant_set[i])
                result_count_included = 0
                top_results = []
                # for result in results_for_this_image[:top_results_considered]:
                for result in results_for_this_image:
                    # if result_count_included > t_rel:
                    #     break

                    if result[0] not in images_included_in_result_set:
                        top_results.append(result)
                        images_included_in_result_set.append(result[0])
                        result_count_included += 1

                # if len(results_for_this_image) > t_rel:
                #     results_for_this_image = results_for_this_image[:t_rel]

                # rel_result_sets[relevant_set[i]] = results_for_this_image
                rel_result_sets[relevant_set[i]] = top_results
                xyz += len(top_results)

            for rel_id, image_list in rel_result_sets.items():
                temp = []
                for im in image_list:
                    if im not in irrelevant_related_images:
                        temp.append(im)

                rel_result_sets[rel_id] = temp

            feature_matrix = []
            new_image_ids = []
            for k, v in rel_result_sets.items():
                for result in v:
                    h5_image_idx = result[1]
                    img_id = result[0]
                    if img_id not in irrelevant_set:
                        new_image_ids.append(img_id)
                        feature_vector = data_cm[h5_image_idx]
                        feature_matrix.append(feature_vector)

            predictions = clf.predict(feature_matrix)

            # predicted 'relevant' by decision tree
            final_image_ids = []
            final_feature_matrix = []

            for new_image_id, feature_vector, prediction in zip(new_image_ids, feature_matrix, predictions):
                # relevant
                if prediction == 0:
                    final_image_ids.append(new_image_id)
                    final_feature_matrix.append(feature_vector)

            # distance of each new image ID(all relevant) from query image
            final_image_distances = [np.linalg.norm(feature_vector - query_image_features)
                                     for feature_vector in final_feature_matrix]

            # sorting relevant image IDs based on distance from query image
            ranked_image_ids = [x for _, x in sorted(zip(final_image_distances, final_image_ids), key=lambda pair: pair[0])]
            feature_matrix = [x for _, x in sorted(zip(final_image_distances, final_feature_matrix), key=lambda pair: pair[0])]
            final_image_distances.sort()

            final_image_scores = []
            for final_image_distance in final_image_distances:
                if final_image_distance != 0:
                    final_image_scores.append(1 / final_image_distance)
                else:
                    final_image_scores.append(1)

            print_ranked_results_v2(query_image_id, ranked_image_ids, final_image_scores, t_rel)
            # print('New results based on relevace feedback(renewd query formation):\nImage ID : Score')
            # for ranked_image_id, final_image_score in zip(ranked_image_ids[:t_rel], final_image_scores[:t_rel]):
            #     print(f'{ranked_image_id} : {final_image_score}')

            provide_rel_feedback = input("Do you want to provide relevance feedback? [Enter 'yes' or 'no']\n")

            if 'yes' in provide_rel_feedback:
                # get user input for relevant and irrelevant images
                # relevant_set += input_rel_irrelevant_image_ids()
                # irrelevant_set += input_rel_irrelevant_image_ids(is_relevant = False)
                relevant_set_temp, _ = input_rel_irrelevant_image_ids()  # is a list of image_id's
                relevant_set += relevant_set_temp
                irrelevant_set_temp, relevant_set = input_rel_irrelevant_image_ids(is_relevant=False,
                                                                              relevant_set=relevant_set)
                irrelevant_set += irrelevant_set_temp

    # PPR based relevance feedback system
    elif rel_feed_model == "3":
        number_of_edges = 10
        number_of_edges_irr = 7
        t_rel = 20
        top_irr_to_discard = 5

        provide_rel_feedback = 'yes'

        while 'yes' in provide_rel_feedback:
            irrelevant_related_images = []
            irr_result_sets = {}
            for i in range(len(irrelevant_set)):
                results_for_this_image = lsh_index.query_image(irrelevant_set[i])
                if len(results_for_this_image) > top_irr_to_discard:
                    results_for_this_image = results_for_this_image[:t_rel]

                irr_result_sets[irrelevant_set[i]] = results_for_this_image
                # for result in results_for_this_image:
                #     top_results.append(result)
                    # irrelevant_related_images.append(result[0])
            irr_feature_matrix = []
            irr_image_ids = []
            for k, v in irr_result_sets.items():
                for result in v:
                    h5_image_idx = result[1]
                    img_id = result[0]
                    if img_id not in irr_image_ids:
                        irr_image_ids.append(img_id)
                        feature_vector = data_cm[h5_image_idx]
                        irr_feature_matrix.append(feature_vector)

            similarity_dict_2 = dimension_reduction(np.array(irr_feature_matrix), number_of_edges_irr, selected_model,
                                                   dim_reduction_tech,
                                                   irr_image_ids)
            irrel_img_ppr = ppr(similarity_dict_2, irrelevant_set, len(irr_feature_matrix))
            # Found after multiple run that we will have around 3 irrelevant images for each irrelevant marked
            # images by the user, so we remove len of user irrelevant set * 3
            irrel_img_ppr = irrel_img_ppr[:(len(irrelevant_set)*3)]

            for ir_ppr in irrel_img_ppr:
                irrelevant_related_images.append(ir_ppr[0])

            # TODO: make sure to have query image vector as well
            # best we can include all the results from
            # do lsh using irrelevant as well and remove some images from actual image query
            # and remove unwanted from actual result set
            rel_result_sets = {}
            images_included_in_result_set = []
            for i in range(len(relevant_set)):
                results_for_this_image = lsh_index.query_image(relevant_set[i])
                result_count_included = 0
                top_results = []
                for result in results_for_this_image:
                    if result_count_included > t_rel:
                        break

                    if result[0] not in images_included_in_result_set:
                        top_results.append(result)
                        images_included_in_result_set.append(result[0])
                        result_count_included += 1

                # if len(results_for_this_image) > t_rel:
                #     results_for_this_image = results_for_this_image[:t_rel]

                # rel_result_sets[relevant_set[i]] = results_for_this_image
                rel_result_sets[relevant_set[i]] = top_results

            feature_matrix = []
            new_image_ids = []
            for k, v in rel_result_sets.items():
                for result in v:
                    h5_image_idx = result[1]
                    img_id = result[0]
                    if img_id not in irrelevant_set and img_id not in irrelevant_related_images:
                        new_image_ids.append(img_id)
                        feature_vector = data_cm[h5_image_idx]
                        feature_matrix.append(feature_vector)

            similarity_dict_ = dimension_reduction(np.array(feature_matrix), number_of_edges, selected_model, dim_reduction_tech,
                                                   new_image_ids)
            rel_img_ppr = ppr(similarity_dict_, relevant_set, len(feature_matrix))
            ppr_ranked_images = []

            for ppr1 in rel_img_ppr:
                ppr_ranked_images.append(ppr1[0])

            counter = 0
            user_specified_t = len(similarity_list)
            similarity_list = []
            for ppr1 in rel_img_ppr:
                if counter == user_specified_t:
                    break
                if ppr1[0] in ppr_ranked_images:
                    img_id = ppr1[0]
                    img_score = ppr1[1]
                    h5_image_idx = lsh_index.image_id_map[img_id]
                    temp = (img_id, h5_image_idx, img_score)
                    similarity_list.append(temp)
                    counter += 1

            # print the new result set obtained after including relevance feedback
            print_ranked_results(similarity_list, relevant_set)

            provide_rel_feedback = input("Do you want to provide relevance feedback? [Enter 'yes' or 'no']\n")

            if 'yes' in provide_rel_feedback:
                # get user input for relevant and irrelevant images
                # relevant_set += input_rel_irrelevant_image_ids()
                # irrelevant_set += input_rel_irrelevant_image_ids(is_relevant = False)
                relevant_set_temp, _ = input_rel_irrelevant_image_ids()  # is a list of image_id's
                relevant_set += relevant_set_temp
                irrelevant_set_temp, relevant_set = input_rel_irrelevant_image_ids(is_relevant=False,
                                                                                   relevant_set=relevant_set)
                irrelevant_set += irrelevant_set_temp

    # TODO: RFeedback with new query formulation
    elif rel_feed_model == "4":
        provide_rel_feedback = 'yes'

        while 'yes' in provide_rel_feedback:
            relevant_index = [lsh_index.image_id_map[u] for u in relevant_set]
            irrelevant_index = [lsh_index.image_id_map[u] for u in irrelevant_set]

            similarity_index = [u for _, u, _ in similarity_list]

            p_i_relevant = np.sum(binary_data_hog_[relevant_index], axis=0) / len(relevant_index)
            p_i_irrelevant = np.sum(binary_data_hog_[irrelevant_index], axis=0) / len(irrelevant_index)

            ratio = (p_i_relevant + 1e-6) / (p_i_irrelevant + 1e-6)

            # term_values1 = p_i*(1-u_i)/(1-p_i)
            # term_values2 = u_i
            term_values_log = np.log(ratio)
            term_values_log = term_values_log.reshape((-1, 1))
            ranked_list_matrix = binary_data_hog_[similarity_index]

            new_similarity_score_list = np.dot(ranked_list_matrix, term_values_log)
            new_similarity_score_list = new_similarity_score_list.reshape(-1)
            new_ranked_list = []
            for i in range(len(new_similarity_score_list)):
                sim = new_similarity_score_list[i]
                u, v, w = similarity_list[i]
                new_ranked_list.append((u, v, sim))
            new_ranked_list = sorted(new_ranked_list, key=lambda x: x[2], reverse=True)
            similarity_list = new_ranked_list

            print_ranked_results(similarity_list, relevant_set)

            provide_rel_feedback = input("Do you want to provide relevance feedback? [Enter 'yes' or 'no']\n")

            if 'yes' in provide_rel_feedback:
                # get user input for relevant and irrelevant images
                # relevant_set += input_rel_irrelevant_image_ids()
                # irrelevant_set += input_rel_irrelevant_image_ids(is_relevant=False)
                relevant_set_temp, _ = input_rel_irrelevant_image_ids()  # is a list of image_id's
                relevant_set += relevant_set_temp
                irrelevant_set_temp, relevant_set = input_rel_irrelevant_image_ids(is_relevant=False,
                                                                                   relevant_set=relevant_set)
                irrelevant_set += irrelevant_set_temp


if __name__ == "__main__":
    run_task()