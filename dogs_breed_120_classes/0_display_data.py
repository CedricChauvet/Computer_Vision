
import cv2
import scipy as sp
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import scipy.ndimage as spi
import matplotlib.pyplot as plt
# %matplotlib inline
np.random.seed(42)

DATASET_PATH = r'./train/'
LABEL_PATH = r'./labels.csv'

# This function prepares a random batch from the dataset
# selection using loc for a take a specific cell
def load_batch(dataset_df, batch_size = 25):
    batch_df = dataset_df.loc[np.random.permutation(np.arange(0,
                                                              len(dataset_df)))[:batch_size],:]
    return batch_df
    

# plots les images de la base de données en utilsant des images pandas
def plot_batch_2 (images_df, grid_width, grid_height, im_scale_x, im_scale_y):
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(12, 12)
    
    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            ax[i][j].axis('off')
            ax[i][j].set_title(images_df.iloc[img_idx]['breed'][:10])
            image_2 = cv2.imread(DATASET_PATH + images_df.iloc[img_idx]['id']+'.jpg') # iloc gives the index  with respect to id column
            ax[i][j].imshow(image_2)
            img_idx += 1

    # display that image 
    plt.show() 
# load dataset and visualize sample data
dataset_df = pd.read_csv(LABEL_PATH)
batch_df = load_batch(dataset_df, batch_size=36)
plot_batch_2(batch_df, grid_width=6, grid_height=6,
           im_scale_x=64, im_scale_y=64)