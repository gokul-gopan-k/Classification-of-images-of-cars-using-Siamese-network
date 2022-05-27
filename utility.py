import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def make_pairs(images, labels):
    "Function that create positive and negative pair of images  and corresponding labels" 
    "for siamese model"
    
    pairImages = []
    pairLabels = []
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the same class label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels not  to the current label and 
        # randomly pick an image corresponding to a label not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images 
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type 
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss 
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss

def plot_training(H):
    "Function to plot loss and accuracy after training"
    
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
