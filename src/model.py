from tensorflow.keras.models import Model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input


def create_model():
    base_model = ConvNeXtTiny(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    # 512 is only the initial number of units - it may be changed later
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
