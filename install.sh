# detection packages
conda create -n cryolo -c conda-forge -c anaconda python=3.6 pyqt=5 cudnn=7.1.2 numpy==1.14.5 cython wxPython==4.0.4 intel-openmp==2019.4 pip=20.2.3
conda activate cryolo
pip install 'cryolo[gpu]'
conda deactivate

# classification packages
conda create -n classify python=3.6 -y
conda activate classify
pip install tensorflow==2.3.0
pip install mrcfile
conda install matplotlib -y
pip install keras-tuner 
conda deactivate
