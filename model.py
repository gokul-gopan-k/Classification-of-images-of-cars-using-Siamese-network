from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv2D,Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from config import CONFIG

def build_siamese_model(inputShape, embeddingDim=CONFIG.Embedding_dim):  
    "Function that create siamese model"
    "Returns embedding as per dimension"
    
    inputs = Input(inputShape)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    model = Model(inputs, outputs)
    
    return model
