## CryoGridML (in development)
CryoGridML can detect cryo-EM grid square meshes from cryo-EM grid maps (patch format) and classify the detected squares into high- and low quality categories. Training networks and hyperparameter optimization are also included.

## Updates
24.11.
- added preprocessing: patch format -> image format -> downscaling

14.11.
- added classification tuning
- added classification training
- added f1-metric
- fixed usage of GPUs instead of CPUs
- added in classification results:
    - best parameters saved
    - config saved
- added architectures
    - efficientNetB5
    - efficientNetB6
    - efficientNetB7
    - ResNet -> ResNetV2
    - InceptionResNetV2
    - NASNetLarge
    - DenseNet121
    - DenseNet169
    - DenseNet201
- changed
    - changed augment-filling from "nearest" to "reflect"
    - tuning objective now val_acc instead of loss


## Requirements
To run CryoGridML *Nvidia* GPUs are required. We also need *CUDA* and *cuDNN* to be installed. Make sure that the installed versions are compatible with *tensorflow* (https://www.tensorflow.org/install/source#gpu), e.g.:

- CUDA (10.1)
- cuDNN (7.6.4)
- Tensorflow (2.3)

A package manager such as *Anaconda/ Miniconda* is also required, e.g.:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda init
```

## Install
```bash
mkdir cryoGridML
cd cryoGridML
source install.sh
```

## Run 
Start CryoGridML with the following commands:
```bash
source setEnv.sh
source main.sh -p data/predict/2020-12-14.mrc #not working right now
```

Flags for main.sh:
- -pd: predict detector
- -pc: predict classifier
- -p : predict (detector + classifier)
- -td: train detector
- -tc: train classifier (not working right now)

Results can be found in:
- detection/results
- classification/results/


## Configuration
Configuration settings are located at:
- *classification/config_classifier.json*
- *detection/config_cryolo.json*

Data for prediction (grid maps) are here:
- *data/predict/*

Data for training (grid squares) are here:
- *data/trainClassifier/*
- *data/trainDetector/*