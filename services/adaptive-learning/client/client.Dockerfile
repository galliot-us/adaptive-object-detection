#This container will install TensorFlow Object Detection API and its dependencies in the /model/research/object_detection directory

FROM tensorflow/tensorflow:1.15.0-gpu-py3
VOLUME /repo
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y git protobuf-compiler python3-tk vim wget

RUN pip install Cython && \
    pip install contextlib2 && \
    pip install pillow && \
    pip install lxml && \
    pip install jupyter && \
    pip install matplotlib


WORKDIR /repo/client
#ENTRYPOINT ["bash", "./train_student.sh"]
#CMD ["-c", "/repo/applications/adaptive-learning/configs/iterdet.ini"]
