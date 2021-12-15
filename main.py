import argparse
from src.Detector import Detector
from src.Classifier import Classifier
from os.path import isfile, splitext
from pathlib import Path


def main():
    # set cmd line arguments
    parser = argparse.ArgumentParser(description='Detect cryoEM grid squares.')
    parser.add_argument(
        '-p', '--predict', help='Predict cryoEM grid maps. Requires .mrc-file as argument')
    parser.add_argument(
        '-td', '--train_detector', help='Train cryoEM grid square detector.', action='store_true')
    parser.add_argument(
        '-tc', '--train_classifier', help='Train cryoEM grid square classifier.', action='store_true')
    args = parser.parse_args()

    # assert mrc-file
    fileMRC = args.predict
    if fileMRC:
        if not isfile(fileMRC):
            raise ValueError("Path input file does not exist.")
        if splitext(fileMRC)[1] != ".mrc":
            raise ValueError("Argument is not an .mrc file.")

    # init
    detector = Detector()
    classifier = Classifier()

    # run
    if args.train_detector:
        print("START TRAINING DETECTOR.")
        detector.train()
        print("FINISH TRAINING DETECTOR.")

    if args.train_classifier:
        # classifier.train()
        # classifier.useTrainedModels = True
        # classifier.predict()
        pass

    if args.predict:
        print("START PREDICTION.")
        detector.initMRC(fileMRC=fileMRC)
        detector.preprocess1()
        detector.preprocess2()  # TODO: way too slow
        detector.preprocess3()
        detector.predict()
        classifier.predict(dataName=Path(detector.fileResized).stem)
        print("FINISH PREDICTION.")


if __name__ == "__main__":
    main()
