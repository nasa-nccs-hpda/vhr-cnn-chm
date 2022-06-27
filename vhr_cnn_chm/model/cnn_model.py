import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate, Input, UpSampling2D, add
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def get_2d_cnn_tf(input_size=(256, 256, 3), filters=(16, 32, 64), regression=False, chanDim=-1):

    # define the model input
    inputs = Input(shape=input_size)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        
        # if this is the first CONV layer, initialize with input
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # regression execution
    if regression:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs=inputs, outputs=x, name="SimpleRegression_2dCNN")

    # return the CNN
    return model


def unet_batchnorm_regression(nclass=19, input_size=(256, 256, 8), weight_file=None,
                   kr=l2(0.0001), maps=[64, 128, 256, 512, 1024]
                   ):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size, name='Input')

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    n1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(n1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    n2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(n2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    n3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(n3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    n4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(n4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    n6 = BatchNormalization()(u6)
    u6 = concatenate([n6, n4])
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    n7 = BatchNormalization()(u7)
    u7 = concatenate([n7, n3])
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    n8 = BatchNormalization()(u8)
    u8 = concatenate([n8, n2])
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    n9 = BatchNormalization()(u9)
    u9 = concatenate([n9, n1], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    # actv = 'softmax'
    # if nclass == 1:
    #    actv = 'sigmoid'

    # c10 = Conv2D(nclass, (1, 1), activation=actv, kernel_regularizer=kr)(c9)
    # model = Model(inputs=inputs, outputs=c10, name="UNetBatchNorm")
    model = Model(inputs=inputs, outputs=c9, name="UNetBatchNormRegression")

    if weight_file:
        model.load_weights(weight_file)
    return model

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    regression_cnn_2d = get_2d_cnn_tf(
        input_size=(125, 125, 8), filters=(16, 32, 64), regression=False)
    regression_cnn_2d.summary()

    regression_cnn_unet = unet_batchnorm_regression(
        nclass=1, input_size=(32, 32, 8), kr=l2(0.0001), maps=[8, 16, 32, 64, 128])
    regression_cnn_unet.summary()
