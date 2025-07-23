'''
File: prepare_dataset.py
Author: Guido Di Federico
Description: Used for loading a diffusers "dataset"-like dataset of geomodels and save it as .jpg images in a folder
Note: requires Python package "datasets" by HuggingFace to open dataset
'''

# Import packages
import os
import numpy as np
import cv2
from datasets import Dataset

# Set directories for dataset and .jpg images folder
dataset_dir = './data/diffusers_dataset/'
imgs_dir = './data/imgs/'

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
    
# Load diffusers dataset and transform into NumPy
diffusers_dataset = Dataset.load_from_disk(dataset_dir)['image']
geomodels_numpy_rot = np.array([np.array(image) for image in diffusers_dataset]) 
geomodels_numpy = np.rot90(np.flip(geomodels_numpy_rot, axis=2), k=-1, axes=(1, 2))

# Transform every NumPy geomodel into .jpg image and save in folder
for i in range(len(diffusers_dataset)):
    img_path = os.path.join(imgs_dir, f"image_{i}.jpeg")
    cv2.imwrite(img_path, geomodels_numpy[i,:,:])

