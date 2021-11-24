import subprocess
import sys
import argparse
import tensorflow as tf
import mrcfile
import numpy as np
from glob import glob
from pathlib import Path
from os.path import join, isfile, splitext


class Detector:
    """ 
        Detection system for cryoEM grid squares based on the crYOLO network implementation.
        In: cryoEM grid maps  
        Out: cryoEM grid square coordinates
    """

    def __init__(self):
        self.pathConfig = "detection/config_cryolo.json"
        self.pathModel = "detection/cryolo_model.h5"
        self.pathBlend = "detection/dataBlend"
        self.pathResized = "detection/dataResized"
        self.pathResults = "detection/results/"

        self.resizeDim = [3740,3822] # [7480,7644]
        self.fileMRC = ""
        self.filePL = ""
        self.fileBlend = ""
        self.fileResized = ""

        self.warmUp = "5"
        self.gpus = "0 1"
        self.confidenceThreshold = "0.3"
        self.batch_size = "2" # changed from 3 to 2 due to GPU memory limitations
        self.gpu_fraction = "1.0"
        self.distance = "10"
        self.minsize = "20"

    def predict(self):
        program = ["cryolo_predict.py"]
        arguments = ["-c", self.pathConfig,
                     "-w", self.pathModel,
                     "-i", self.pathResized,
                     "-o", self.pathResults,
                     "-t", self.confidenceThreshold,
                     "-g", self.gpus,
                     "-pbs", self.batch_size,
                     "--gpu_fraction", self.gpu_fraction,
                     "-d", self.distance,
                     "--minsize", self.minsize,
                     ]

        process = subprocess. run(
            program + arguments, stdout=sys.stdout, stderr=subprocess.PIPE)

        if process.returncode != 0:
            sys.exit("Cryolo prediction error: " + process.stderr.decode())

    def train(self):
        program = ["cryolo_train.py"]
        arguments = ["-c", self.pathConfig,
                     "-w", self.warmUp,
                     "-g", self.gpus,
                     ]

        process = subprocess. run(
            program + arguments, stdout=sys.stdout, stderr=subprocess.PIPE)

        if process.returncode != 0:
            sys.exit("Cryolo training error: " + process.stderr.decode())

    def preprocess1(self, fileMRC):
        fileNew = join(self.pathBlend, Path(fileMRC).stem + ".pl")
        self.fileMRC = fileMRC
        self.filePL = fileNew
        program = ["extractpieces"]
        arguments = ["-input", fileMRC, "-output", fileNew]

        process = subprocess. run(
            program + arguments, stdout=sys.stdout, stderr=subprocess.PIPE)

        if process.returncode != 0:
            sys.exit("Preprocessing error: " + process.stderr.decode())

    def preprocess2(self):
        program = ["blendmont"]
        fileNew = join(self.pathBlend, Path(self.fileMRC).stem + "_blend.mrc")
        fileBase = join(self.pathBlend, Path(self.fileMRC).stem + "_blend")
        self.fileBlend = fileNew
        arguments = ["-v",
                     "-imi", self.fileMRC,
                     "-pli", self.filePL,
                     "-imo", fileNew,
                     "-roo", fileBase
                     ]

        process = subprocess. run(
            program + arguments, stdout=sys.stdout, stderr=subprocess.PIPE)

        if process.returncode != 0:
            sys.exit("Preprocessing error: " + process.stderr.decode())

    def preprocess3(self):
        self.fileBlend = join(self.pathBlend, Path(
            self.fileMRC).stem + "_blend.mrc")  # TODO: remove
        with mrcfile.open(self.fileBlend) as mrc:
            image = mrc.data
            image = np.expand_dims(image, axis=3)
            image = np.expand_dims(image, axis=0)
            imageNew = tf.image.resize_bilinear(image, self.resizeDim)
            imageNew = imageNew.eval(session=tf.Session())
            imageNew = np.squeeze(imageNew)
            self.fileResized = join(self.pathResized, Path(
                self.fileMRC).stem + "_resized.mrc")
            with mrcfile.new(self.fileResized, overwrite=True) as mrc2:
                mrc2.set_data(imageNew)


def main():
    # set cmd line arguments
    parser = argparse.ArgumentParser(description='Detect cryoEM grid squares.')
    parser.add_argument(
        '-p', '--predict', help='Predict cryoEM grid maps.')
    parser.add_argument(
        '-pd', '--predict_detector', help='Detect cryoEM grid squares. Requires .mrc-file as argument')
    parser.add_argument(
        '-td', '--train_detector', help='Train cryoEM grid square detector.', action='store_true')
    parser.add_argument(
        '-pc', '--predict_classifier', help='Classify cryoEM grid squares.', action='store_true')
    parser.add_argument(
        '-tc', '--train_classifier', help='Train cryoEM grid square classifier.', action='store_true')
    args = parser.parse_args()

    # assert argument: mrc-file
    fileMRC = args.predict_detector  # TODO: set to predict only
    if fileMRC:
        if not isfile(fileMRC):
            raise ValueError("Path input file does not exist.")
        if splitext(fileMRC)[1] != ".mrc":
            raise ValueError("Argument is not an .mrc file.")

    # run detection
    detector = Detector()
    if args.train_detector:
        print("START TRAINING DETECTOR.")
        detector.train()
        print("FINISH TRAINING DETECTOR.")
    if args.predict_detector or args.predict:
        print("START DETECTION.")
        detector.preprocess1(fileMRC)
        # detector.preprocess2()
        detector.preprocess3()
        detector.predict()
        print("FINISH DETECTION.")


if __name__ == "__main__":
    main()
