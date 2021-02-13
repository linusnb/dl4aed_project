from enum import Enum
import tensorflow as tf


class ModelType(Enum):
    """
    Enumeration of different models
    """
    HENNEQUIN = 1
    BASIC_CNN = 2


class ModelBuilder:
    """
    Model buiilder
    """
    def __init__(self, modeltype: ModelType, input_shape: [int, int],
                 classes: list):
        self.modeltype = modeltype
        self.input_shape = input_shape
        self.classes = classes

        # build model
        self.model = self.build_model()

    def get_model(self):
        """
        returns build model
        """
        return self.model

    def build_model(self):
        """
        build model depending on model type, classes and input shape
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=self.input_shape))
        model.add(tf.keras.layers.BatchNormalization())

        if self.modeltype == ModelType.HENNEQUIN:
            model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.GlobalMaxPool2D())
            model.add(tf.keras.layers.Dense(256, activation="relu"))
            model.add(tf.keras.layers.GaussianDropout(0.25))
            model.add(tf.keras.layers.Dense(256, activation="relu"))
            model.add(tf.keras.layers.GaussianDropout(0.25))
            model.add(tf.keras.layers.Dense(len(self.classes),
                                            activation="softmax"))
            return model

        elif self.modeltype == ModelType.BASIC_CNN:
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.GaussianDropout(0.25))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.GaussianDropout(0.25))
            model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
            model.add(tf.keras.layers.GlobalMaxPool2D())
            model.add(tf.keras.layers.Dense(len(self.classes),
                                            activation="sigmoid"))
            return model
        else:
            print('No valid modeltype')
