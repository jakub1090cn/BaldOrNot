from tensorflow.keras.models import Model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from constants import IMG_LEN, NUM_CHANNELS, NUM_DENSE_UNITS


def create_model():
    base_model = ConvNeXtTiny(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS),
    )
    for layer in base_model.layers:
        layer.trainable = False
    inputs = Input(shape=(IMG_LEN, IMG_LEN, NUM_CHANNELS))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(NUM_DENSE_UNITS, activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
