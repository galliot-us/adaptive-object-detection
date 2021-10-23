# docker can be installed on the dev board following these instructions:
# https://docs.docker.com/install/linux/docker-ce/debian/#install-using-the-repository , step 4: arm64
# 1) build: docker build -f Dockerfile -t "neuralet/jetson-nano:tf-ssd-to-trt" .
# 2) run: docker run -it --runtime nvidia --privileged --network host -v /PATH_TO_DOCKERFILE_DIRECTORY/:/repo neuralet/jetson-nano:tf-ssd-to-trt

#FROM nvcr.io/nvidia/l4t-base:r32.6.1

FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf1.15-py3
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

VOLUME /repo

RUN apt-get update && apt-get install -y python3-pip pkg-config zip gnupg

RUN python3 -m pip install pip==20.1

RUN apt-get install -y python3-numpy 

RUN python3 -m pip install pycuda

RUN apt-get install -y vim git
RUN printf 'deb https://repo.download.nvidia.com/jetson/common r32 main\ndeb https://repo.download.nvidia.com/jetson/t210 r32 main' > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

COPY ./trusted-keys /tmp/trusted-keys
RUN apt-key add /tmp/trusted-keys
RUN apt-get update 
RUN apt-get install -y tensorrt
RUN apt-get install -y libnvinfer6 libnvinfer-dev python3-libnvinfer python3-libnvinfer-dev
RUN apt-get install -y graphsurgeon-tf uff-converter-tf
RUN pip3 install protobuf
RUN apt-get install -y pkg-config libhdf5-100 libhdf5-dev
RUN apt-get install -y python3-h5py python3-opencv
RUN pip3 install wget pillow nvidia-pyindex

COPY ./exporters/libflattenconcat.so.6 /opt/libflattenconcat.so 
RUN apt update && apt install -y libtcmalloc-minimal4
WORKDIR /repo
ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"
