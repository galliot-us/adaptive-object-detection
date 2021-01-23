FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

VOLUME  /repo
WORKDIR /repo

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget pillow

