# import sklearn
# import matplotlib
# import tifffile
import cv2
import os
import numpy as np
import random
import re

# This file load the original Fluo-N2DH-SIM+, resize it to 768x768 samples, concatenates folders,
# and devide them randomly into train, validation and test sets : 60% train set, 20% validation set, 20% test set
random.seed(1212)



def data_partition(images_path):
    files_list = list_files(images_path)

    X_train = list()
    X_val = list()
    X_test = list()


    #test file is validation indices files and test2 file is test indices file
    os.chdir(r'C:\Users\noamb\PycharmProjects\Volcani\arcface-pytorch\data\facebank')
    train_file = open('train.txt', 'w')
    validation_file = open(r'validation.txt', 'w')
    test_file = open(r'test.txt', 'w')

    num_of_examples = len(files_list)
    idx = np.arange(num_of_examples)
    #idx += 1
    np.random.shuffle(idx)

    train_set_size = int(num_of_examples * 0.7)
    train_idx = idx[0: train_set_size]

    val_set_size = int(num_of_examples * 0.2)
    val_idx = idx[train_set_size: train_set_size+val_set_size]

    test_set_size = int(num_of_examples * 0.1)
    test_idx = idx[train_set_size+val_set_size : train_set_size+val_set_size+test_set_size]

    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    for i1 in train_idx:
        X_train.append(str('data/images/')+files_list[i1])
        train_file.write('data/images/'+files_list[i1] + '\n')


    for i2 in val_idx:
        X_val.append(str('data/images/')+files_list[i2])
        validation_file.write('data/images/' + files_list[i2] + '\n')


    for i3 in test_idx:
        X_test.append(str('data/images/')+files_list[i3])
        test_file.write('data/images/' + files_list[i3] + '\n')

    train_file.close()
    validation_file.close()
    test_file.close()



def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

# Load
path_01 = r'C:\Users\noamb\PycharmProjects\Volcani\arcface-pytorch\data\facebank'
data_partition(path_01)

print('111')


