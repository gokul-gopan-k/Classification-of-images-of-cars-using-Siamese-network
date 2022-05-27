from tensorflow.keras.layers import Input
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from model import build_siamese_model
from config import CONFIG
from utility import make_pairs,euclidean_distance,contrastive_loss,plot_training
from prepare_data import preprocess_data

def training_model():
    "Function for training"
    
    #Get train and valdation data
    x_train, y_train = preprocess_data(mode = "train")
    x_val, y_val     =  preprocess_data(mode = "validation")
    
    # prepare the positive and negative pairs
    print("[INFO] preparing positive and negative pairs...")
    (pairTrain, labelTrain) = make_pairs(x_train, y_train)
    (pairTest, labelTest) = make_pairs(x_val, y_val)

    imgA = Input(shape=(CONFIG.img_width, CONFIG.img_height, CONFIG.channel_width))
    imgB = Input(shape=(CONFIG.img_width, CONFIG.img_height, CONFIG.channel_width))
    featureExtractor = build_siamese_model((CONFIG.img_width, CONFIG.img_height, CONFIG.channel_width))
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    
    distance = euclidean_distance([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance)
    
    checkpoint = ModelCheckpoint(
        'UNET_model',
        monitor='val_loss',
        verbose=1, 
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
    )

    early_stopping = EarlyStopping(
        patience=5,
        min_delta=0.0001,
        restore_best_weights=True,
    )

    # compile the model
    print("[INFO] compiling model...")
    model.compile(loss=contrastive_loss, optimizer="adam",
        metrics=["accuracy"])
    # train the model
    print("[INFO] training model...")
    history = model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        batch_size=CONFIG.BATCH_SIZE, 
        epochs=CONFIG.EPOCHS,verbose=1)

    plot_training(history)
    
    return model
