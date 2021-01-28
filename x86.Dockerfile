FROM tensorflow/tensorflow:2.2.2-py3

VOLUME  /repo
WORKDIR /repo

RUN apt update && apt install -y libgl1-mesa-glx zip vim

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget pillow

