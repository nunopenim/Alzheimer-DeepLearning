import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
import keras
from tensorflow import keras as tkr

from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import optimizers

# Data preprocessing - Here the main processing of data happens

base_dir = "training_dataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)

os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)

WORK_DIR = './dataset/'
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = tkr.preprocessing.image.ImageDataGenerator(rescale=1. / 255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)
train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)

# Data balancing - Using SMOTE technique

train_data, train_labels = train_data_gen.next()
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state=42)

sm = SMOTE(random_state=42)

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)
train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Model Design - CNN

pooling = tkr.layers.MaxPooling2D(2, 2)
fine_dropout = tkr.layers.Dropout(0.1)
coarse_dropout = tkr.layers.Dropout(0.5)
flat_them_layers = tkr.layers.Flatten()

layer1 = tkr.layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (176,176,3))
layer2 = tkr.layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same')
layer3 = tkr.layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same')
layer4 = tkr.layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same')
out_layer = tkr.layers.Dense(units = 4, activation = 'softmax')

model = tkr.models.Sequential()
model.add(layer1)
model.add(pooling)
model.add(fine_dropout)
model.add(layer2)
model.add(pooling)
model.add(fine_dropout)
model.add(layer3)
model.add(pooling)
model.add(fine_dropout)
model.add(layer4)
model.add(pooling)
model.add(coarse_dropout)
model.add(flat_them_layers)
model.add(out_layer)

print('done with no errors')
