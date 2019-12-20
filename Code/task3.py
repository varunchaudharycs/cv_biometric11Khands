from image_features.colormoments import ColorMoment
from image_features.HOG import HOG
from image_features.FeatureModel import *
from utils.cli_options import *
import ast
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import cv2 as cv
from math import ceil
import os
import pickle
import pagerank as pagerank
import scipy.sparse as sparse

CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = CURRENT_DIR + os.sep + '..' + os.sep + 'Outputs'


class ImageSimilarity:
    def __init__(self, image_id, similarity):
        self.image_id = image_id
        self.similarity = similarity


def display_ppr(img_ppr, K, folder_path):
    # create grid to display images appropriately
    num_of_cols_in_plot_grid = 5
    num_of_rows_in_plot_grid = int(ceil(K / num_of_cols_in_plot_grid))
    f, axes = plt.subplots(num_of_rows_in_plot_grid, num_of_cols_in_plot_grid, squeeze=False, figsize=(14, 7))
    plt.subplots_adjust(bottom=0.0, left=0.07, right=0.9, top=0.9, wspace=0.1, hspace=0.35)

    # set the title according to the type
    title = 'Personalized Page Ranking'
    f.canvas.set_window_title(title)

    counter = 0
    for i in range(num_of_rows_in_plot_grid):
        for j in range(num_of_cols_in_plot_grid):
            if counter >= K:
                axes[i][j].axis('off')
                continue
            else:
                if counter < len(img_ppr):
                    img_path = folder_path + os.sep + 'Hand_' + img_ppr[counter][0] + '.jpg'
                    img_id = img_ppr[counter][0]

                    img_bgr = cv.imread(img_path)
                    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
                    axes[i][j].imshow(img)
                    axes[i][j].set_title('Img Id %s: %s\nPPR: %f' % (counter + 1, img_id, img_ppr[counter][1]))
                    axes[i][j].axis('off')
                else:
                    axes[i][j].axis('off')
            counter += 1
    plt.show()


def dimension_reduction(X, k, K, selected_model, dim_reduction_tech, image_feature_model, input_images):
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

    # STORE THE SIMILARITY DICT IN H5py or pickle object beforehand
    # Find the similarity
    similarity_dict = {}
    lf_len = len(latent_features)
    for i in range(lf_len):
        for j in range(lf_len):
            if i == j:
                continue

            dist = image_feature_model.euclidean_distance(list(latent_features[i]), list(latent_features[j]))
            sim = round((1 / dist) * 1000, 4)
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


def run_task():
    # selected_model = choose_feature_model()
    selected_model = "3"
    # dim_reduction_tech = choose_dim_reduction_tech()
    dim_reduction_tech = "1"
    folder_path = input_data_set_folder_path()
    # image_label = choose_image_label()
    k = int(input('Enter the value of k (outgoing edges): '))
    K = int(input('Enter the value of K (most dominant images): '))

    i = 0
    print('\nEnter 3 image IDs:')
    images = []
    while i < 3:
        img_id = input('Image ID {0}: '.format(str(i+1)))
        images.append(img_id)
        i += 1

    np.random.seed(81)
    if selected_model == "1":
        image_feature_model = ColorMoment(folder_path)
        input_images = image_feature_model.get_data_set_files()
        feature_matrix = image_feature_model.extract_feature_feature_vectors()
        X = np.array(feature_matrix)
        similarity_dict = dimension_reduction(X, k, K, selected_model, dim_reduction_tech, image_feature_model, input_images)
        related_images = []
        weights = []
        personalize = []
        image_id_map = {}
        temp = 0
        for k, v in similarity_dict.items():
            image_id_map[k] = temp
            temp += 1

        for k, v in similarity_dict.items():
            if k in images:
                personalize.append(1/(len(images)))
            else:
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
        pr = pagerank.personalizedPageRank(G, personalize, c=0.20, allowedDiff=1e-9, maxIters=200)
        pr_sorted = np.argsort(pr)
        pr_sorted = pr_sorted[::-1]

        img_ppr = []
        print('\nPersonalized Page ranking are:\n-------------------------------------------')
        for t in range(0, K):
            id = pr_sorted[t]
            score = pr[id]
            for image_id, node_id in image_id_map.items():
                if node_id == id:
                    # print('\n')
                    print('Image ID ' + str(t + 1) + ': ' + image_id + ' : ' +  str(score))

                    # Store the information to display it using matplot
                    temp = [image_id, score]
                    img_ppr.append(temp)

        display_ppr(img_ppr, K, folder_path)

    elif selected_model == "3":
        image_feature_model = HOG(folder_path)
        input_images = image_feature_model.get_data_set_files()
        feature_matrix = image_feature_model.extract_feature_feature_vectors()
        X = np.array(feature_matrix)
        similarity_dict = dimension_reduction(X, k, K, selected_model, dim_reduction_tech, image_feature_model, input_images)
        related_images = []
        weights = []
        personalize = []
        image_id_map = {}
        temp = 0
        for k, v in similarity_dict.items():
            image_id_map[k] = temp
            temp += 1

        for k, v in similarity_dict.items():
            if k in images:
                personalize.append(1/(len(images)))
            else:
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
        pr = pagerank.personalizedPageRank(G, personalize, c=0.20, allowedDiff=1e-9, maxIters=200)
        pr_sorted = np.argsort(pr)
        pr_sorted = pr_sorted[::-1]

        img_ppr = []
        print('\nPersonalized Page ranking are:\n-------------------------------------------')
        for t in range(0, K):
            id = pr_sorted[t]
            score = pr[id]
            for image_id, node_id in image_id_map.items():
                if node_id == id:
                    # print('\n')
                    print('Image ID ' + str(t + 1) + ': ' + image_id + ' : ' +  str(score))

                    # Store the information to display it using matplot
                    temp = [image_id, score]
                    img_ppr.append(temp)

        display_ppr(img_ppr, K, folder_path)


if __name__ == "__main__":
    run_task()