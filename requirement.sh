#!/bin/bash
apt update -y && apt upgrade -y

apt-get install libgl1-mesa-glx -y

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install scikit-image
pip install pandas
pip install tqdm
