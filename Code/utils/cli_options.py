"""
Common utility functions defined in this file.
Use these functins anywhere by importing the file at the top of the desired location as follows:
'from functions import *'

@author: Abhishek Mugal
"""

from image_features.FeatureModel import *
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)

def input_data_set_folder_path(task=None, type=None):
    # CURRENT_DIR = os.path.dirname(__file__)

    folder_dict = {
        1: os.path.join(CURRENT_DIR, '..' + os.sep + '..' + os.sep, 'Inputs' + os.sep + 'Labelled' + os.sep + 'Set1'),
        2: os.path.join(CURRENT_DIR, '..' + os.sep + '..' + os.sep, 'Inputs' + os.sep + 'Labelled' + os.sep + 'Set2'),
        3: os.path.join(CURRENT_DIR, '..' + os.sep + '..' + os.sep, 'Inputs' + os.sep + 'Unlabelled' + os.sep + 'Set 1'),
        4: os.path.join(CURRENT_DIR, '..' + os.sep + '..' + os.sep, 'Inputs' + os.sep + 'Unlabelled' + os.sep + 'Set 2')
    }

    # Set default folder path
    folder_path = folder_dict[1]

    while True:
        if type == 'training':
            selected_dataset = int(input('\nChoose the testing dataset folder:\n1. Unlabelled Set 1\n2. Unlabelled Set 2\n3. Other\nOption? '))
            selected_dataset += 2
            if selected_dataset == 3 or selected_dataset == 4 or selected_dataset == 5:
                if selected_dataset == 5:
                    folder_path = input('\nEnter custom folder path: ').strip()
                else:
                    folder_path = folder_dict[selected_dataset]
                break
            else:
                print('Please choose correct option.\n')
        else:
            selected_dataset = int(input('\nChoose the dataset folder:\n1. Labelled Set 1\n2. Labelled Set 2\n3. Other\nOption? '))
            if selected_dataset == 1 or selected_dataset == 2 or selected_dataset == 3:
                if selected_dataset == 3:
                    folder_path = input('\nEnter custom folder path: ').strip()
                else:
                    folder_path = folder_dict[selected_dataset]
                break
            else:
                print('Please choose correct option.\n')

    pre_configured_path = os.path.abspath(folder_path)

    if task != None:
        return pre_configured_path.strip(), selected_dataset
    else:
        return pre_configured_path.strip()


# function to choose a model based on the implementation
# @return: model number
#           1. Color Moments, 2. LBP, 3. HOG, 4. SIFT
def choose_feature_model():
    model = ''
    while True:
        # take model as input
        model = input('Please select a Feature Model'
                      ' (1 / 2 / 3 / 4):\n1. Color moments\n2. LBP\n3. HOG\n4. SIFT\n\nOption? ')
        if model == "1" or model == "2" or model == "3" or model == "4":
            return model
        else:
            print('\nYou have entered an invalid option value!')


def choose_classifier():
    while True:
        classifier = input('Please select a classifier'
                      ' (1 / 2 / 3):\n1. SVM classifer\n2. Decision-tree classifier\n3. PPR-based classifier\nOption? ')
        if classifier == "1" or classifier == "2" or classifier == "3":
            return classifier
        else:
            print('\nYou have entered an invalid option value!')


def choose_relevance_feedback_model():
    while True:
        rel_feedback = input('Please select a relevance feedback model'
                      ' (1 / 2 / 3 / 4):\n1. SVM based relevance feedback system'
                           '\n2. Decision-tree based relevance feedback system'
                           '\n3. PPR-based relevance feedback system'
                           '\n4. Probabilistic relevance feedback system'
                            '\nOption? ')
        if rel_feedback == "1" or rel_feedback == "2" or rel_feedback == "3" or rel_feedback == "4":
            return rel_feedback
        else:
            print('\nYou have entered an invalid option value!')

# function to choose a dimension reduction technique
# @return: choosen dimension reduction technique number
#           1. PCA, 2. SVD, 3. NMF, 4. LDA
def choose_dim_reduction_tech():
    dim_reduction_tech = ''
    while True:
        # take model as input
        dim_reduction_tech = input('Please select a dimension reduction technique (1 / 2 / 3 / 4):\n1. PCA\n2. SVD\n3. NMF\n4. LDA\n\nOption? ')
        if dim_reduction_tech == "1" or dim_reduction_tech == "2" or dim_reduction_tech == "3" or dim_reduction_tech == "4":
            return dim_reduction_tech
        else:
            print('\nYou have entered an invalid option value!')


def choose_valueof_k():
    return int(input('Enter the value of k (number of latent space): '))


def choose_valueof_m():
    return int(input('Enter the value of m (number of images to search): '))


# function to get the feature model name based on the model number
# @params: model (1, 2, 3, 4) in string format
# @return: model name
def get_feature_model(model):
    if model == '': return False
    else:
        if model == '1': return 'Color_Moments'
        elif model == '2': return 'LBP'
        elif model == '3': return 'HOG'
        elif model == '4': return 'SIFT'


# function to get the dimentionality reduction technique name based on the corresponding number
# @params: model (1, 2, 3, 4) in string format
# @return: model name
def get_dim_reduction_tech(dim_reduction_tech):
    if dim_reduction_tech == '': return False
    else:
        if dim_reduction_tech == '1': return 'PCA'
        elif dim_reduction_tech == '2': return 'SVD'
        elif dim_reduction_tech == '3': return 'NMF'
        elif dim_reduction_tech == '4': return 'LDA'


# function to choose a image label
# @return: ImageLabel
#      1. left-hand, 2. right-hand, 3. dorsal 4. palmar, 5. with accessories, 6. without accessories, 7. male, 8. female
def choose_image_label():
    while True:
        print('Choose option to specify label '
              '\n1. left-hand, '
              '\n2. right-hand, '
              '\n3. dorsal '
              '\n4. palmar, '
              '\n5. with accessories, '
              '\n6. without accessories, '
              '\n7. male, '
              '\n8. female')
        image_label_choice = input('\nEnter the label choice: ')
        # image_label = ImageLabel.UNSPECIFIED

        if image_label_choice == '1':
            return ImageLabel.LEFT_HAND
        elif image_label_choice == '2':
            return ImageLabel.RIGHT_HAND
        elif image_label_choice == '3':
            return ImageLabel.DORSAL
        elif image_label_choice == '4':
            return ImageLabel.PALMAR
        elif image_label_choice == '5':
            return ImageLabel.WITH_ACCESSORIES
        elif image_label_choice == '6':
            return ImageLabel.WITHOUT_ACCESSORIES
        elif image_label_choice == '7':
            return ImageLabel.MALE
        elif image_label_choice == '8':
            return ImageLabel.FEMALE
        else:
            print('You have entered an invalid option value!\n')


def label_to_label_text(image_label_enum):
    if image_label_enum == ImageLabel.LEFT_HAND:
        return 'left'
    elif image_label_enum == ImageLabel.RIGHT_HAND:
        return 'right'
    elif image_label_enum == ImageLabel.DORSAL:
        return 'dorsal'
    elif image_label_enum == ImageLabel.PALMAR:
        return 'palmar'
    elif image_label_enum == ImageLabel.WITH_ACCESSORIES:
        return 'with_acc'
    elif image_label_enum == ImageLabel.WITHOUT_ACCESSORIES:
        return 'without_acc'
    elif image_label_enum == ImageLabel.MALE:
        return 'male'
    elif image_label_enum == ImageLabel.FEMALE:
        return 'female'


def get_dorsal_palmar_images():
    data_frame = pd.read_csv(os.path.join(CURRENT_DIR, '..' + os.sep, '..' + os.sep, 'Metadata', 'HandInfo.csv'))
    aspect_of_hand = 'aspectOfHand'
    dorsal_right = 'dorsal right'
    dorsal_left = 'dorsal left'
    palmar_right = 'palmar right'
    palmar_left = 'palmar left'
    image_name_column = 'imageName'
    dorsal_images = []
    palmar_images = []

    for i in range(data_frame.shape[0]):
        row = data_frame.iloc[i]
        if row[aspect_of_hand] == dorsal_left or row[aspect_of_hand] == dorsal_right:
            img_name = row[image_name_column]
            file_arr = img_name.split('_')[1]
            img_id = file_arr.split('.')[0]
            dorsal_images.append(img_id)
        elif row[aspect_of_hand] == palmar_left or row[aspect_of_hand] == palmar_right:
            img_name = row[image_name_column]
            file_arr = img_name.split('_')[1]
            img_id = file_arr.split('.')[0]
            palmar_images.append(img_id)

    return set(dorsal_images), set(palmar_images)
