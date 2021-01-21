FROM tensorflow/tensorflow:2.2.2-gpu-py3

VOLUME  /repo
WORKDIR /repo

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget pillow

