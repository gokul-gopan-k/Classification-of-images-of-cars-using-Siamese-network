import numpy as np
from prepare_data import preprocess_data
from training import training_model
from config import CONFIG
import os

#get test data
x_test, x_val = preprocess_data(mode = "test")

#test sample
test_in = x_test[0]

#sample from each class
sample = x_val[:11]

probs = []

#train model
model = training_model()

# loop over all image pairs and get most similar class
for i in sample:
    imageA = test_in
    imageB = i
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)
    preds = model.predict([imageA, imageB])
    probs.append(preds[0][0])
class_out = np.argmin(probs) 

labels = os.listdir(CONFIG.data_path)
print("Input image is most similar to class:" labels[class_out])
