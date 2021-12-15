conda create -n cryoGridML python=3.6 -y
conda activate classify
pip install tensorflow==2.3.0
pip install mrcfile
conda install matplotlib -y
pip install keras-tuner 
pip install opencv-python
conda deactivate
