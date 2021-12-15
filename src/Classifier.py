import os
import json
import argparse
import numpy as np
import mrcfile
import cv2
from tensorflow import keras
from datetime import datetime
from src.Data import Data
from src.Model import Model
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
        self.pathModel = "classification/model_classification.h5"
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
        self.saveGridMaps = self.config["predict"]["saveGridMaps"]

    def predict(self, dataName):
        # make directory
        print("START CLASSIFICATION.")
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.pathNow = os.path.join(self.pathResultsPred, now)
        os.makedirs(self.pathNow, exist_ok=True)

        # get models
        if self.useTrainedModels:
            self.models = []  # TODO
        else:
            self.data = Data(
                type="predict", pathResultsPred=self.pathNow, name=dataName)
            self.models = [
                Model(nr=0, path=self.pathModel, data=self.data)
            ]

        # predict
        for m in self.models:
            predictions = m.model.predict(x=self.data.data_predict, verbose=1)
            predictions_max = np.max(predictions, axis=-1)
            predictions_i = np.argmax(predictions, axis=-1)
            predictions_label = np.vectorize(
                lambda x: self.data.classes[x])(predictions_i)

            # save results
            filePath = os.path.join(self.pathNow, f"predict_{m.name}.txt")
            f = open(filePath, "w")
            f.write("x,y,prediction,confidence\n")
            for i, (x, y) in enumerate(self.data.coords):
                pred = predictions_label[i]
                conf = str(predictions_max[i])
                f.write(','.join([str(x), str(y), pred,  conf + "\n"]))
            f.close()

            # save gridmap.jpeg with bounding boxes
            if self.saveGridMaps:
                with mrcfile.open(self.data.dataMRC) as mrc:
                    # normalize
                    imgBox = np.copy(mrc.data)
                    imgBox = np.stack((imgBox,)*3, axis=-1)  # 1->3channels
                    imgBox = cv2.normalize(
                        imgBox, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    imgBox = imgBox.astype(np.uint8)
                    
                    # draw rectangle + text
                    d = int(self.data.imgSize/2)
                    i = zip(self.data.coords, predictions_i, predictions_max)
                    for (x, y), pred, conf in i:
                        conf = str(round(conf, 4))
                        color = (36, 255, 12) if pred == 0 else (255, 0, 0)
                        up, down, left, right = (y+d, y-d, x-d, x+d)
                        imgBox = cv2.rectangle(imgBox, pt1=(left, down), pt2=(right, up),
                                               color=color, thickness=5)
                        cv2.putText(
                            imgBox, conf, (left+10, down+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9, color=color, thickness=2)
                    
                    dataFile = os.path.join(self.pathNow, "gridmap.jpg")
                    plt.imsave(dataFile, imgBox)

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
