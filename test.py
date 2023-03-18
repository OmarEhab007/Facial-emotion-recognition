import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import cv2 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from IPython.display import SVG, Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# from livelossplot import PlotLossesKeras
# from tensorflow.keras.utils import np_utils
print("Tensorflow version:", tf.__version__)




df = pd.read_csv('train.csv')
# df.head()


mg_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48,48,1).astype('float32'))
img_array = np.stack(img_array, axis=0)
img_array.shape 

le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = to_categorical(img_labels)
img_labels.shape 

le_name_mappig = dict(zip(le.classes_, le.transform(le.classes_)))
le_name_mappig

x_train, x_valid, y_train, y_valid = train_test_split(
    img_array,
    img_labels,
    shuffle=True,
    stratify=img_labels,
    test_size=0.2,
    random_state=42
)

del df
del img_array 
del img_labels

print(x_train.shape, x_valid.shape , y_train.shape, y_valid.shape)


#Normalizing arrays, as neural networks are very sensitive to unnormalized data.
x_train = x_train / 255.
x_valid = x_valid / 255.

img_width = x_train.shape[1]
img_height = x_train.shape[2]
img_depth = x_train.shape[3]
num_classes = y_train.shape[1]




