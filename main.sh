#!/bin/bash

# run detection
conda activate cryolo
python ./src/Detector.py "$@"
conda deactivate

# run classification
conda activate classify
python ./src/Classifier.py "$@"
conda deactivate

# exit
trap 'conda deactivate' EXIT
