import subprocess
import sys
import argparse
import mrcfile
import numpy as np
from glob import glob
from pathlib import Path
from os.path import join, isfile, splitext
from tensorflow import keras
from PIL import Image


class Detector:
    """ 
        Detection system for cryoEM grid squares based on the crYOLO network implementation.
        In: cryoEM grid maps  
        Out: cryoEM grid square coordinates
    """

    def __init__(self):
        self.pathConfig = "detection/config_cryolo.json"
        self.pathModel = "detection/cryolo_model.h5"
        self.pathBlend = "detection/1_blend"
        self.pathResized = "detection/2_resized"
        self.pathResults = "detection/3_results/"

        self.resizeDim = [7644, 7480]
        self.warmUp = "5"
        self.gpus = "0 1"
        self.confidenceThreshold = "0.3"
        self.batch_size = "3"
        self.gpu_fraction = "1.0"
        self.distance = "10"
        self.minsize = "20"

    def initMRC(self, fileMRC):
        self.fileMRC = fileMRC
        self.fileName = Path(self.fileMRC).stem
        self.filePL = join(self.pathBlend, self.fileName + ".pl")
        self.fileBlend = join(self.pathBlend, self.fileName + "_blend")
        self.fileBlendMRC = join(self.pathBlend, self.fileName + "_blend.mrc")
        self.fileResized = join(self.pathResized, self.fileName + "r.mrc")

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

    def runProcess(self, program, arguments):
        process = subprocess. run(
            program + arguments, stdout=sys.stdout, stderr=subprocess.PIPE)
        if process.returncode != 0:
            sys.exit("Process error: " + process.stderr.decode())

    def preprocess1(self):
        print("1. START EXTRACTPIECES.")
        program = ["extractpieces"]
        arguments = ["-input", self.fileMRC, "-output", self.filePL]
        self.runProcess(program, arguments)

    def preprocess2(self):
        print("2. START BLENDMONT.")
        program = ["blendmont"]
        arguments = ["-v",
                     "-imi", self.fileMRC,
                     "-pli", self.filePL,
                     "-imo", self.fileBlendMRC,
                     "-roo", self.fileBlend
                     ]
        self.runProcess(program, arguments)

    def preprocess3(self):
        print("3. START DOWNSAMPLING.")
        with mrcfile.mmap(self.fileBlendMRC, mode='r+') as mrc:
            image = mrc.data
            imagePIL = Image.fromarray(image)
            imagePIL = imagePIL.resize(self.resizeDim, resample=Image.BILINEAR)
            image = np.array(imagePIL, dtype=np.int16)
            with mrcfile.new(self.fileResized, overwrite=True) as mrc2:
                mrc2.set_data(image)
                mrc2.close()
            mrc.close()

    def predict(self):
        print("4. START CRYOLO PREDICTION.")
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
        self.runProcess(program, arguments)
