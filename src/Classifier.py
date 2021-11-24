import os
import json
import argparse
import numpy as np
from tensorflow import keras
from datetime import datetime
from Data import Data
from Model import Model
import matplotlib.pyplot as plt

class Classifier:
    """
        Classification system for cryoEM grid squares based on Deep Neural Networks.
        In: cryoEM grid square coordinates
        Out: Label: "good" or "bad"
    """

    def __init__(self):
        self.nr = int("0")  # TODO: overwrite
        self.pathResultsPred = "classification/results/predict"
        self.pathTrain = "classification/results/train"
        self.pathNow = None
        self.data = None
        self.models = []

        self.config = json.load(open('classification/config_classifier.json'))
        self.modelConfig = self.config["train"]["models"][self.nr]
        self.useTrainedModels = False
        self.session_name = self.config["train"]["session_name"]
        self.batch_size = self.config["train"]["models"][self.nr]["batch_size"]
        self.name = self.config["train"]["models"][self.nr]["name"]
        self.epochs_tuner = self.config["train"]["models"][self.nr]["epochs_tuner"]
        self.epochs = self.config["train"]["models"][self.nr]["epochs"]

    def predict(self):
        print("START CLASSIFICATION.")
        if self.useTrainedModels:
            self.models = []  # TODO
        else:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.pathNow = os.path.join(self.pathResultsPred, now)
            os.makedirs(self.pathNow, exist_ok=True)
            self.data = Data(type="predict", pathResultsPred=self.pathNow)
            self.models = [
                Model(nr=0, path="classification/model_classification.h5",
                      data=self.data)
            ]
        plt.imsave(self.pathNow+"/test.jpg", self.data.data_predict[10], cmap='gray')
        self.data.data_predict = np.expand_dims(self.data.data_predict, axis=3)

            
        for m in self.models:
            # predict
            predictions = m.model.predict(x=self.data.data_predict, verbose=1)
            predictions_max = np.max(predictions, axis=-1)
            predictions_i = np.argmax(predictions, axis=-1)
            predictions_label = np.vectorize(
                lambda x: self.data.classes[x])(predictions_i)

            # save results
            filePath = os.path.join(self.pathNow, f"predictions_{m.nr}.txt")
            f = open(filePath, "w")
            f.write("x,y,prediction,confidence,file\n")
            index = 0  # index for prediction list
            for j, fileCoords in enumerate(self.data.coords):
                for i, (x, y) in enumerate(fileCoords):
                    f.write(
                        ','.join([str(x), str(y), predictions_label[index],
                                  str(predictions_max[index]), self.data.fileNames[j] + ".mrc\n"]))
                    index += 1
            f.close()

    def train(self):
        # create dir, save config
        print("START TRAINING CLASSIFIER.")
        self.data = Data(type="train")
        self.pathNow = os.path.join(
            self.pathTrain, self.session_name, self.name)
        os.makedirs(self.pathNow, exist_ok=True)
        pathConfig = os.path.join(self.pathNow, "config.json")
        json.dump(self.modelConfig, open(pathConfig, 'w'))

        # search optimal hyperparameters
        model = Model(nr=self.nr, data=self.data)
        tuner = model.tuner
        tuner.search(
            x=self.data.train_data,
            y=self.data.train_labels,
            validation_data=(self.data.val_data, self.data.val_labels),
            batch_size=self.batch_size,
            epochs=self.epochs_tuner,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=10)],
            verbose=2,
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        pathParam = os.path.join(self.pathNow, "best_parameters.json")
        json.dump(best_hps.values, open(pathParam, 'w'))

        # train model
        model = tuner.hypermodel.build(best_hps)
        model.summary()
        hist = model.fit(
            x=self.data.train_data,
            y=self.data.train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.data.val_data, self.data.val_labels),
            verbose=2,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    os.path.join(self.pathNow, "model.h5"),
                    monitor="val_accuracy",
                    verbose=2,
                    save_best_only=True,
                    mode="auto",
                )
            ],
        )
        pathHist = os.path.join(self.pathNow, "history.json")
        json.dump(hist.history, open(pathHist, 'w'))

    def evaluate(self, models):
        print("START EVALUATION.")


def main():
    # set cmd line arguments
    parser = argparse.ArgumentParser(
        description='Clasify cryoEM grid squares.')
    parser.add_argument(
        '-p', '--predict', help='Predict cryoEM grid maps.')
    parser.add_argument(
        '-pc', '--predict_classifier', help='Classify cryoEM grid squares.', action='store_true')
    parser.add_argument(
        '-tc', '--train_classifier', help='Train cryoEM grid square classifier.', action='store_true')
    parser.add_argument(
        '-pd', '--predict_detector', help='Detect cryoEM grid squares.')
    parser.add_argument(
        '-td', '--train_detector', help='Train cryoEM grid square detector.', action='store_true')
    args = parser.parse_args()

    # run classification
    classifier = Classifier()
    if args.train_classifier:
        classifier.train()
        classifier.useTrainedModels = True
        # classifier.predict()
    if args.predict_classifier or args.predict:
        classifier.predict()


if __name__ == "__main__":
    main()
