"""
Création des data de train, val et tests
"""
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import img_to_array, load_img


# from keras.preprocessing.image import load_img
import pandas as pd

def prepa():
     data_labels = pd.read_csv('./dog-breed-identification/labels.csv') 
     train_folder = './dog-breed-identification/train/'
     data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["id"] + ".jpg" ), 
                                                  axis=1)


     # load dataset
     train_data = np.array([img_to_array(load_img(img, target_size=(299, 299)))
                              for img in data_labels['image_path'].values.tolist()
                         ]).astype('float32')


     target_labels = data_labels['breed']
     # create train and test datasets
     x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels, 
                                                       test_size=0.3, 
                                                       stratify=np.array(target_labels), 
                                                       random_state=42)

     # create train and validation datasets
     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                       test_size=0.15, 
                                                       stratify=np.array(y_train), 
                                                       random_state=42)

     print('Initial Dataset Size:', train_data.shape)
     print('Initial Train and Test Datasets Size:', x_train.shape, x_test.shape)
     print('Train and Validation Datasets Size:', x_train.shape, x_val.shape)
     print('Train, Test and Validation Datasets Size:', x_train.shape, x_test.shape, x_val.shape)

     # hot one encode
     y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).to_numpy()
     y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).to_numpy()
     y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).to_numpy()

     print("hot one:",y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape)
     return x_train,y_train_ohe, x_val, y_val_ohe