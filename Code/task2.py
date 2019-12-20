import ast
import numpy as np

from image_features.colormoments import ColorMoment
from image_features.SIFT import SIFT
from image_features.HOG import HOG
from image_features.LBP import LBP
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from utils.cli_options import *
from utils.distance_measures import euclidean_distance
from utils.distance_measures import chi_squared_distance
from utils.similarity_measures import dot_pdt_similarity
from utils.similarity_measures import cosine_similarity

import os, sys, webbrowser, random, time, argparse
import pandas as pd
from scipy.spatial import distance

import csv

CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(CURRENT_DIR, '..' + os.sep, 'Outputs', 'Task_2')
OUTPUT_STATIC_DIR = os.path.join(CURRENT_DIR, '..' + os.sep, 'Outputs', 'static')


def run_task():

    # USER INPUTS
    clusters = int(input('c (number of clusters) : '))
    # labelled_folder = input("Folder : ")
    # unlabelled_folder = input("Classify : ")

    labelled_folder, dataset_type = input_data_set_folder_path(task=4)
    unlabelled_folder = input_data_set_folder_path(type='training')

    labelled_set = labelled_folder[-1]
    unlabelled_set = unlabelled_folder[-1]

    # INPUT FOLDERS & METADATAS
    labelled_folder_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                        'Inputs', 'Labelled', 'Set' + labelled_set)
    labelled_metadata_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                          'Inputs', 'labelled_set' + labelled_set + '.csv')
    labelled_metadata = pd.read_csv(labelled_metadata_path, index_col = "id")

    unlabelled_folder_path = os.path.join(CURRENT_DIR, '..' + os.sep,
                                          'Inputs', 'Unlabelled', 'Set ' + unlabelled_set)

    #################################### LABELLED ##############################################


    labelled_feature_model = FeatureModel(labelled_folder_path)
    labelled_images = labelled_feature_model.get_data_set_files()

    d_list = []
    p_list = []

    d_count = 0
    p_count = 0

    for image in labelled_images:
        aspect = labelled_metadata[labelled_metadata['imageName'] == 'Hand_' + image.image_id + '.jpg']['aspectOfHand'].iloc[0]
        if 'palmar' in aspect:
            p_list.append(image)
            p_count += 1
        if 'dorsal' in aspect:
            d_list.append(image)
            d_count += 1

    #print('Dorsal count - ', d_count)
    #print('Palmar count - ', p_count)
    #create_image_subject_map(image_list, METADATA)

    labelled_feature_model = ColorMoment(labelled_folder_path)

    d_feature_matrix = labelled_feature_model.extract_feature_feature_vectors_list(d_list)
    p_feature_matrix = labelled_feature_model.extract_feature_feature_vectors_list(p_list)

    x_dorsal = np.array(d_feature_matrix)
    x_palmar = np.array(p_feature_matrix)

    #################################### DORSAL ##############################################

    print('Clustering - Dorsal ...')

    centroids_dorsal = {}
    # (#centroid, image IDs)
    dorsal_clusters = {}

    for i in range(clusters):
        centroids_dorsal[i] = x_dorsal[i]

    iterations = 0

    while True:
        iterations += 1
        classifications_dorsal = {}

        for i in range(clusters):
            classifications_dorsal[i] = []

            dorsal_clusters[i] = []

        for featureset, image in zip(x_dorsal, d_list):
            distances = [np.linalg.norm(featureset - centroids_dorsal[centroid]) for centroid in centroids_dorsal]
            classification = distances.index(min(distances))
            classifications_dorsal[classification].append(featureset)

            dorsal_clusters[classification].append(image.image_id)

        is_changed = False

        for classification in classifications_dorsal:
            new_centroid = np.average(classifications_dorsal[classification], axis = 0)

            if not np.array_equal(centroids_dorsal[classification], new_centroid):
                is_changed = True
                centroids_dorsal[classification] = new_centroid

        if not is_changed:
            break

    DORSAL_CLUSTERS_PATH = os.path.join(OUTPUT_DIR,
                                        'dorsal_clusters' + str(clusters) +
                               '_labelledset' + labelled_set +
                               '_unlabelledset' + unlabelled_set +
                               '.csv')

    with open(DORSAL_CLUSTERS_PATH, 'w', newline = "") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dorsal_clusters.items():
            writer.writerow([key, value])

        print('DORSAL (#Centroid, Image IDs) info saved in file - ', DORSAL_CLUSTERS_PATH)

    DORSAL_CENTROIDS_PATH = os.path.join(OUTPUT_DIR,
                                         'dorsal_centroids' + str(clusters) +
                                        '_labelledset' + labelled_set +
                                        '_unlabelledset' + unlabelled_set +
                                        '.csv')

    with open(DORSAL_CENTROIDS_PATH, 'w', newline = "") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in centroids_dorsal.items():
            writer.writerow([key, value])

        print('DORSAL (#Centroid, Centroid value) info saved in file - ', DORSAL_CENTROIDS_PATH)

    print('Clustering - Dorsal ... DONE.')

    #print(f'Iterations - {iterations}')
    # print('Centroid # : # of images')
    # for i in range(0, clusters):
    #     print('{} : {}'.format(i, len(classifications_dorsal[i])))

    #################################### PALMAR ##############################################

    print('Clustering - Palmar ...')

    centroids_palmar = {}
    # (#centroid, image IDs)
    palmar_clusters = {}

    for i in range(clusters):
        centroids_palmar[i] = x_palmar[i]

    iterations = 0

    while True:
        iterations += 1
        classifications_palmar = {}

        for i in range(clusters):
            classifications_palmar[i] = []

            palmar_clusters[i] = []

        for featureset, image in zip(x_palmar, p_list):
            distances = [np.linalg.norm(featureset - centroids_palmar[centroid]) for centroid in centroids_palmar]
            classification = distances.index(min(distances))
            classifications_palmar[classification].append(featureset)

            palmar_clusters[classification].append(image.image_id)

        is_changed = False

        for classification in classifications_palmar:
            new_centroid = np.average(classifications_palmar[classification], axis = 0)

            if not np.array_equal(centroids_palmar[classification], new_centroid):
                is_changed = True
                centroids_palmar[classification] = new_centroid

        if not is_changed:
            break

    PALMAR_CLUSTERS_PATH = os.path.join(OUTPUT_DIR,
                                         'palmar_clusters' + str(clusters) +
                                        '_labelledset' + labelled_set +
                                        '_unlabelledset' + unlabelled_set +
                                        '.csv')

    with open(PALMAR_CLUSTERS_PATH, 'w', newline = "") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in palmar_clusters.items():
            writer.writerow([key, value])

        print('PALMAR (#Centroid, Image IDs) info saved in file - ', DORSAL_CLUSTERS_PATH)

    PALMAR_CENTROIDS_PATH = os.path.join(OUTPUT_DIR,
                                         'palmar_centroids' + str(clusters) +
                                         '_labelledset' + labelled_set +
                                         '_unlabelledset' + unlabelled_set +
                                         '.csv')

    with open(PALMAR_CENTROIDS_PATH, 'w', newline = "") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in centroids_palmar.items():
            writer.writerow([key, value])

        print('PALMAR (#Centroid, Centroid value) info saved in file - ', DORSAL_CENTROIDS_PATH)

    print('Clustering - Palmar ... DONE.')

    #print(f'Iterations - {iterations}')
    #print('Centroid # : # of images')
    # for i in range(0, clusters):
    #     print('{} : {}'.format(i, len(classifications_palmar[i])))

    #################################### UNLABELLED ##############################################

    print('Classifying unlabelled images ...')

    unlabelled_feature_model = FeatureModel(unlabelled_folder_path)
    unlabelled_images = unlabelled_feature_model.get_data_set_files()

    unlabelled_feature_model = ColorMoment(unlabelled_folder_path)

    unlabelled_feature_matrix = unlabelled_feature_model.extract_feature_feature_vectors_list(unlabelled_images)
    x_unlabelled = np.array(unlabelled_feature_matrix)

    aspect_dict = {}

    # Used to calculate accuracy of the classification
    dorsal_images, palmar_images = get_dorsal_palmar_images()
    hits = 0

    for featureset, unlabelled_image in zip(x_unlabelled, unlabelled_images):
        print('-' * 30)
        p_distances = [np.linalg.norm(featureset - centroids_palmar[centroid]) for centroid in centroids_palmar]
        d_distances = [np.linalg.norm(featureset - centroids_dorsal[centroid]) for centroid in centroids_dorsal]
        if min(p_distances) <= min(d_distances):
            aspect_dict[unlabelled_image.image_id] = 'palmar'
            nearest_centroid = p_distances.index(min(p_distances))
            centroid_distance = min(p_distances)
            if unlabelled_image.image_id in palmar_images:
                hits += 1
        else:
            aspect_dict[unlabelled_image.image_id] = 'dorsal'
            nearest_centroid = d_distances.index(min(d_distances))
            centroid_distance = min(d_distances)
            if unlabelled_image.image_id in dorsal_images:
                hits += 1

        print(f'Image ID : {unlabelled_image.image_id}\nPredicted label : {aspect_dict[unlabelled_image.image_id]}\n'
              f'Nearest centroid : c{nearest_centroid}, Distance : {centroid_distance}')

    print('\nClassifying unlabelled images ... DONE.')

    accuracy = (hits / len(unlabelled_images)) * 100
    print(f'\nAccuracy = {accuracy}%')

    OUTPUT_PATH = os.path.join(OUTPUT_DIR,
                               'clusters' + str(clusters) +
                               '_labelledset' + labelled_set +
                               '_unlabelledset' + unlabelled_set +
                               '.csv')

    with open(OUTPUT_PATH, 'w', newline = "") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in aspect_dict.items():
            writer.writerow([key, value])

        print('(ImageID,AspectOfHand) info saved in file - ', OUTPUT_PATH)

#     visualization of clusters

    main_html_file_content = '<!DOCTYPE html><html><head><title>Cluster Visualization</title><style></style></head> <body><table>'

    cluster_content_prefix = '<!DOCTYPE html><html><head><title>Cluster Visualization</title><script type="text/javascript" ' \
                             'src="' + OUTPUT_STATIC_DIR + os.sep + 'vivagraph.js">' \
                             '<style> div{font-size:120%;}</style></script><script type="text/javascript">function main () {' \
                             'var graph = Viva.Graph.graph();' \
                             'graph.addNode(\'centroid\', \'' + str(OUTPUT_STATIC_DIR + os.sep) + 'black-dot.png\');'

    cluster_content_suffix = """
    var layout = Viva.Graph.Layout.forceDirected(graph, {
                springLength : 220
            });
    var graphics = Viva.Graph.View.svgGraphics();
                             graphics.node(function(node) {
                             var url = node.data;
                             return Viva.Graph.svg('image').attr('width', 80)
                     .attr('height', 80)
                     .link(url);
            });""" + 'graphics.placeNode(function(nodeUI, pos) {' \
                             'nodeUI.attr(\'x\', pos.x - 18).attr(\'y\', pos.y - 18);' \
                                                                  '});var renderer = Viva.Graph.View.renderer(graph, {' \
                             'layout : layout,'\
                             'graphics : graphics});' \
                             'renderer.run();}</script>' \
                             '<style type="text/css" media="screen">' \
                             'html, body, svg { width: 100%; height: 100%;}' \
                             '</style></head><body onload=\'main()\'></br><div>'

    cluster_content_suffix2 = '</div></body></html>'

    # dorsal clusters visualization
    dorsal_html_file = OUTPUT_DIR + os.sep + 'dorsal_clusters.html'
    dorsal_html_file_content = ''
    dorsal_html_file_content += main_html_file_content
    counter = 0
    for cluster_name, images in dorsal_clusters.items():
        # if counter % 2 == 0:
        if counter != 0:
            dorsal_html_file_content += '</tr>'
        dorsal_html_file_content += '<tr>'

        cluster_html_file = OUTPUT_DIR + os.sep + 'dor_test' + str(counter) + '.html'
        cluster_html_file_content = ''
        cluster_html_file_content += cluster_content_prefix
        images_html = ''
        counter2 = 0
        for img in images:
            if counter2 != 0:
                images_html += ', '
            images_html += img
            cluster_html_file_content += ' graph.addNode(\'' + img +'\', \'' + os.path.abspath(labelled_folder) + os.sep + 'Hand_' + img + '.jpg\');'
            cluster_html_file_content += ' graph.addLink(\'centroid\', \'' + img + '\');'
            counter2 += 1

        cluster_html_file_content += cluster_content_suffix
        cluster_html_file_content += 'Cluster ID: ' + str(cluster_name)
        cluster_html_file_content += '</br>'
        cluster_html_file_content += 'Image IDs: ' + images_html
        cluster_html_file_content += cluster_content_suffix2

        f2 = open(cluster_html_file, "w")
        f2.write(cluster_html_file_content)
        f2.flush()
        f2.close()

        # dorsal_html_file_content += '<td><iframe src="' + cluster_html_file + '" width="" height="600"></iframe></td>'
        dorsal_html_file_content += '<td><iframe src="' + cluster_html_file + '" onload="this.width=screen.width-300; this.height=screen.height-100;" ></iframe></td>'

        counter += 1

    dorsal_html_file_content += '</table></body></html>'
    f1 = open(dorsal_html_file, "w")
    f1.write(dorsal_html_file_content)
    f1.flush()
    f1.close()

    # palmar cluster visualization
    palmar_html_file = OUTPUT_DIR + os.sep + 'palmar_clusters.html'
    palmar_html_file_content = ''
    palmar_html_file_content += main_html_file_content
    counter = 0
    for cluster_name, images in palmar_clusters.items():
        # if counter % 2 == 0:
        if counter != 0:
            dorsal_html_file_content += '</tr>'
        palmar_html_file_content += '<tr>'

        cluster_html_file = OUTPUT_DIR + os.sep + 'palm_test' + str(counter) + '.html'
        cluster_html_file_content = ''
        cluster_html_file_content += cluster_content_prefix
        images_html = ''
        counter2 = 0
        for img in images:
            if counter2 != 0:
                images_html += ', '
            images_html += img
            cluster_html_file_content += ' graph.addNode(\'' + img + '\', \'' + os.path.abspath(
                labelled_folder) + os.sep + 'Hand_' + img + '.jpg\');'
            cluster_html_file_content += ' graph.addLink(\'centroid\', \'' + img + '\');'
            counter2 += 1

        cluster_html_file_content += cluster_content_suffix
        cluster_html_file_content += 'Cluster ID: ' + str(cluster_name)
        cluster_html_file_content += '</br>'
        cluster_html_file_content += 'Image IDs: ' + images_html
        cluster_html_file_content += cluster_content_suffix2
        f2 = open(cluster_html_file, "w")
        f2.write(cluster_html_file_content)
        f2.flush()
        f2.close()

        # dorsal_html_file_content += '<td><iframe src="' + cluster_html_file + '" width="" height="600"></iframe></td>'
        palmar_html_file_content += '<td><iframe src="' + cluster_html_file + '" onload="this.width=screen.width-300; this.height=screen.height-200;" ></iframe></td>'

        counter += 1

    palmar_html_file_content += '</table></body></html>'
    f1 = open(palmar_html_file, "w")
    f1.write(palmar_html_file_content)
    f1.flush()
    f1.close()

    webbrowser.open('file://' + os.path.realpath(dorsal_html_file))
    webbrowser.open('file://' + os.path.realpath(palmar_html_file))


if __name__ == "__main__":
    print('Task 2 ...')
    run_task()
    print('\nTask 2 ... DONE.')
