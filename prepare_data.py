import os
import cv2
import numpy as np
from config import CONFIG
from sklearn.model_selection import train_test_split

def get_data(data_dir):
    "Function to parse the input images and labels"
    
    input_images = []
    input_labels = []
    
    #Get list of lbels
    labels = os.listdir(data_dir)
    
    #Create image and label matrices
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img))[...,::-1] 
                resized_img = cv2.resize(img_array, (img_width, img_height)) 
                input_images.append(resized_img)
                input_labels.append(class_num)
    return input_images,input_labels

def preprocess_data(mode):
    "Fuction to create input for training and testing"
    "Returns values depending on mode is train,validation or test"
    
    data_dir_train= CONFIG.data_path
    data_images, data_labels = get_data(data_dir_train)
    
    #Create train, validation and test splits
    x_train, x_test, y_train, y_test = train_test_split(data_images, data_labels, test_size=1 - CONFIG.train_ratio)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=CONFIG.test_ratio/(CONFIG.test_ratio + CONFIG.validation_ratio))

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    #return as per mode
    if mode == "train":
        return x_train, y_train
    elif mode == "validation":
        return x_val,y_val
    else:
        return x_test, x_val


