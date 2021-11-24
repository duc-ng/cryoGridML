import glob
import os
import mrcfile
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow import keras


class Data ():
    """
        Class managing cryo-em grid square images.
    """

    def __init__(self, type: str, pathResultsPred: str = ""):
        # init configs
        self.config = json.load(open('classification/config_classifier.json'))
        self.saveGridSquares = self.config["predict"]["saveGridSquares"]
        self.imgSize = self.config["data"]["imgSize"]
        self.saveGridMaps = self.config["predict"]["saveGridMaps"]
        self.pathGoodMRC = self.config["data"]["pathGoodMRC"]
        self.pathBadMRC = self.config["data"]["pathBadMRC"]
        self.classes = self.config["data"]["classes"]

        # init other
        self.globResultsStar = "detection/results/STAR/*.star"
        self.globDataPrediction = "data/predict/*.mrc"
        self.pathResultsPred = pathResultsPred
        self.seed = 42
        self.setNr = 0

        # init data
        if type == "predict":
            preds = self.loadDataPrediction()
            self.data_predict = preds[0]
            self.coords = preds[1]
            self.fileNames = preds[2]
        else:
            train = self.loadDataTrain()
            self.train_data = train[0]
            self.train_labels = train[1]
            self.test_data = train[2]
            self.test_labels = train[3]
            self.val_data = train[4]
            self.val_labels = train[5]

    def getName(self, x) -> str:
        base = os.path.basename(x)
        name = base[:base.index('.')]
        return name

    def loadDataPrediction(self):
        # load predicted coordinates
        predFiles = glob.glob(self.globDataPrediction)
        fileNames = [self.getName(x) for x in predFiles]
        starFiles = list(filter(lambda x: self.getName(
            x) in fileNames, glob.glob(self.globResultsStar)))
        coords = []
        for path in starFiles:
            with open(path, "r") as f:
                coord = []
                for i, line in enumerate(f):
                    if i > 8:
                        entries = line.strip().split('\t')
                        floatVals = float(entries[0]), float(entries[1])
                        roundVals = int(floatVals[0]), int(floatVals[1])
                        coord.append(roundVals)
                coords.append(coord)

        # cut grid squares, drop the border ones
        data = []
        for i, path in enumerate(predFiles):
            with mrcfile.open(path) as mrc:
                image = mrc.data
                d = int(self.imgSize/2) # distance from center
                for x, y in coords[i]:
                    if (y-d) >= 0 and (y+d) <= image.shape[0] and (x-d) >= 0 and (x+d) <= image.shape[1]:
                        imgSquare = image[y-d:y+d, x-d:x+d]
                        data.append(imgSquare)
        data = np.array(data)
        print("(Prediction data) min=%.3f, max=%.3f, mean=%.3f, std=%.3f" %
              (data.min(), data.max(), data.mean(), data.std()))

        # save grid square images as .mrc
        if self.saveGridSquares:
            dataFile = os.path.join(self.pathResultsPred, "gridsquares.mrc")
            with mrcfile.new(dataFile) as mrc:
                mrc.set_data(data)

        # save grid map with bounding boxes
        if self.saveGridMaps:
            for i, path in enumerate(predFiles):
                with mrcfile.open(path) as mrc:
                    imgBox = np.copy(mrc.data)
                    d = int(self.imgSize/2)
                    color = np.max(imgBox)
                    for x, y in coords[i]:
                        up, down, left, right = (y+d, y-d, x-d, x+d)
                        imgBox[up, left:right] = color
                        imgBox[down, left:right] = color
                        imgBox[down:up, left] = color
                        imgBox[down:up, right] = color
                    dataFile = os.path.join(
                        self.pathResultsPred, "gridmap_"+fileNames[i]+".jpg")
                    plt.imsave(dataFile, imgBox, cmap='gray')

        # reshape
        # data = np.expand_dims(data, axis=3)
        return data, coords, fileNames

    def loadDataTrain(self) -> Tuple[List, List]:
        # load data
        with mrcfile.open(self.pathGoodMRC) as mrc:
            data_good = mrc.data
            labels_good = len(data_good)*[0]
        with mrcfile.open(self.pathBadMRC) as mrc:
            data_bad = mrc.data
            labels_bad = len(data_bad)*[1]
        data = np.concatenate((data_good, data_bad))
        labels = np.concatenate((labels_good, labels_bad))
        print("(Train data) min=%.3f, max=%.3f, mean=%.3f, std=%.3f" %
              (data.min(), data.max(), data.mean(), data.std()))

        # preprocess
        data = np.expand_dims(data, axis=3)
        labels = keras.utils.to_categorical(
            labels, len(self.classes))  # one-hot-encode

        # shuffle
        np.random.seed(self.seed)
        idRandom = np.random.permutation(len(data))
        data, labels = data[idRandom], labels[idRandom]

        # split data: 10-fold
        def nr(i):
            return (self.setNr + i) % 10
        data_split = np.array(np.split(data, 10))
        label_split = np.array(np.split(labels, 10))

        # train, test, val: 80%, 10%, 10%
        test_data = data_split[nr(0)]
        test_labels = label_split[nr(0)]
        val_data = data_split[nr(1)]
        val_labels = label_split[nr(1)]
        train_data = np.concatenate(data_split[2:10, :]) # TODO
        train_labels = np.concatenate(label_split[2:10, :]) # TODO

        return train_data, train_labels, test_data, test_labels, val_data, val_labels
