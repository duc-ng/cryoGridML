import json
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing


class Model ():
    """
        Class managing neural network models for cryo-em grid square classification.
    """

    def __init__(self, data, nr: int, path: str = "default"):
        self.config = json.load(open('classification/config_classifier.json'))
        self.imgSize = self.config["data"]["imgSize"]
        self.shape = (self.imgSize, self.imgSize, 1)
        self.classes = self.config["data"]["classes"]
        self.session_name = self.config["train"]["session_name"]
        self.nr = nr
        self.data = data

        self.name = self.config["train"]["models"][nr]["name"]
        self.seed = self.config["train"]["models"][nr]["seed"]
        self.flip = self.config["train"]["models"][nr]["flip"]
        self.hTranslate = self.config["train"]["models"][nr]["hTranslate"]
        self.wTranslate = self.config["train"]["models"][nr]["wTranslate"]
        self.rotate = self.config["train"]["models"][nr]["rotate"]
        self.contrast = self.config["train"]["models"][nr]["contrast"]
        self.activation = self.config["train"]["models"][nr]["activation"]
        self.loss = self.config["train"]["models"][nr]["loss"]
        self.tunerType = self.config["train"]["models"][nr]["tunerType"]
        self.epochs_tuner = self.config["train"]["models"][nr]["epochs_tuner"]
        self.optimizer = self.config["train"]["models"][nr]["optimizer"]
        self.layers_nr = self.config["train"]["models"][nr]["layers_nr"]
        self.layers_units = self.config["train"]["models"][nr]["layers_units"]
        self.layers_act = self.config["train"]["models"][nr]["layers_act"]
        self.activation = self.config["train"]["models"][nr]["activation"]
        self.lr = self.config["train"]["models"][nr]["learning_rate"]
        self.architecture = self.config["train"]["models"][nr]["architecture"]

        if (path != "default"):
            self.model = keras.models.load_model(
                path, custom_objects={'f1': self.f1})
        else:
            self.tuner = self.getTuner()

    def getTuner(self):
        strategy = tf.distribute.MirroredStrategy()
        print("Number of GPUs: ", len(tf.config.list_physical_devices('GPU')))
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        tune_args = {
            "hypermodel": self.getTunerModel,
            "objective": "val_accuracy",
            "seed": self.seed,
            "directory": f"classification/results/train/{self.session_name}/{self.name}",
            "project_name": "tuner",
            "distribution_strategy": strategy,
        }
        if self.tunerType == "bayesian":
            tuner = kt.BayesianOptimization(
                max_trials=100,
                **tune_args
            )
        elif self.tunerType == "random":
            tuner = kt.RandomSearch(
                max_trials=100,
                **tune_args
            )
        else:
            tuner = kt.Hyperband(
                max_epochs=self.epochs_tuner,
                factor=3,
                **tune_args
            )
        return tuner

    def getTunerModel(self, hp):
        model = keras.models.Sequential()
        model = self.preprocess(model, hp)
        model = self.getArchitecture(model)
        model = self.postprocess(model, hp)
        return model

    def preprocess(self, model, hp):
        # resize, normalize
        model.add(preprocessing.Resizing(self.imgSize,
                  self.imgSize, input_shape=self.shape))
        _ = hp.Choice("name", values=[self.name])  # for logging
        normalize = preprocessing.Normalization()
        normalize.adapt(self.data.train_data)
        model.add(normalize)
        # contrast
        model.add(preprocessing.RandomContrast(
            factor=hp.Choice("contrast", values=self.contrast),
            seed=self.seed,
        ))
        # flip
        if hp.Choice("flip", values=self.flip):
            model.add(preprocessing.RandomFlip(
                mode="horizontal_and_vertical", seed=self.seed,
            ))
        # height translate
        model.add(preprocessing.RandomTranslation(
            height_factor=hp.Choice(
                "hTranslate", values=self.hTranslate),
            width_factor=0,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=self.seed,
        ))
        # width translate
        model.add(preprocessing.RandomTranslation(
            height_factor=0,
            width_factor=hp.Choice("wTranslate", values=self.wTranslate),
            fill_mode="reflect",
            interpolation="bilinear",
            seed=self.seed,
        ))
        # rotation
        model.add(preprocessing.RandomRotation(
            factor=hp.Choice("rotate", values=self.rotate),
            fill_mode="reflect",
            interpolation="bilinear",
            seed=self.seed,
        ))
        return model

    def postprocess(self, model, hp):
        # top layers
        model.add(keras.layers.GlobalAveragePooling2D())
        for i in range(hp.Int("layers_nr", self.layers_nr[0], self.layers_nr[1])):
            model.add(
                keras.layers.Dense(
                    units=hp.Int("layers_units_" + str(i), min_value=self.layers_units[0],
                                 max_value=self.layers_units[1], step=self.layers_units[2]),
                    activation=hp.Choice("layers_act", values=self.layers_act),
                )
            )
        model.add(keras.layers.Dense(
            len(self.classes), activation=self.activation))
        # optimizer
        opt = hp.Choice("optimizer", values=self.optimizer)
        args_opt = {"learning_rate": hp.Choice("lr", values=self.lr)}
        if opt == "SGD":
            optimizer = keras.optimizers.SGD(
                **args_opt, momentum=0.9, nesterov=True)
        elif opt == "Adam":
            optimizer = keras.optimizers.Adam(**args_opt)
        elif opt == "RMSprop":
            optimizer = keras.optimizers.RMSprop(**args_opt)
        elif opt == "Adamax":
            optimizer = keras.optimizers.Adamax(**args_opt)
        elif opt == "Nadam":
            optimizer = keras.optimizers.Nadam(**args_opt)
        else:
            raise ValueError('Wrong tuner.')
        # compile

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=["accuracy", "Precision", "Recall", "AUC", f1],
        )
        return model

    def f1(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def getArchitecture(self, model):
        args_model = {
            "include_top": False,
            "weights": None,
            "input_shape": self.shape,
            "classes": len(self.classes),
            "classifier_activation": self.activation,
        }
        if self.architecture == "Xception":
            model.add(keras.applications.Xception(**args_model))
        elif self.architecture == "InceptionV3":
            model.add(keras.applications.InceptionV3(**args_model))
        elif self.architecture == "VGG16":
            model.add(keras.applications.VGG16(**args_model))
        elif self.architecture == "VGG19":
            model.add(keras.applications.VGG19(**args_model))
        elif self.architecture == "ResNet50V2":
            model.add(keras.applications.ResNet50V2(**args_model))
        elif self.architecture == "ResNet101V2":
            model.add(keras.applications.ResNet101V2(**args_model))
        elif self.architecture == "ResNet152V2":
            model.add(keras.applications.ResNet152V2(**args_model))
        elif self.architecture == "EfficientNetB5":
            model.add(keras.applications.EfficientNetB5(**args_model))
        elif self.architecture == "EfficientNetB6":
            model.add(keras.applications.EfficientNetB6(**args_model))
        elif self.architecture == "EfficientNetB7":
            model.add(keras.applications.EfficientNetB7(**args_model))
        elif self.architecture == "InceptionResNetV2":
            model.add(keras.applications.InceptionResNetV2(**args_model))
        elif self.architecture == "NASNetLarge":
            model.add(keras.applications.NASNetLarge(**args_model))
        elif self.architecture == "DenseNet121":
            model.add(keras.applications.DenseNet121(**args_model))
        elif self.architecture == "DenseNet169":
            model.add(keras.applications.DenseNet169(**args_model))
        elif self.architecture == "DenseNet201":
            model.add(keras.applications.DenseNet201(**args_model))
        elif self.architecture == "LeNet":
            model.add(keras.layers.Conv2D(
                filters=6,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=self.shape)
            )
            model.add(keras.layers.AveragePooling2D())
            model.add(keras.layers.Conv2D(
                filters=16, kernel_size=(3, 3), activation="relu"))
            model.add(keras.layers.AveragePooling2D())
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(units=120, activation="relu"))
            model.add(keras.layers.Dense(units=84, activation="relu"))
            model.add(keras.layers.Dense(
                units=len(self.classes), activation=self.activation))
        else:
            raise ValueError('Wrong architecture.')
        return model
