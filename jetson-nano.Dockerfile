# docker can be installed on the dev board following these instructions:
# https://docs.docker.com/install/linux/docker-ce/debian/#install-using-the-repository , step 4: arm64
# 1) build: docker build -f Dockerfile -t "neuralet/jetson-nano:tf-ssd-to-trt" .
# 2) run: docker run -it --runtime nvidia --privileged --network host -v /PATH_TO_DOCKERFILE_DIRECTORY/:/repo neuralet/jetson-nano:tf-ssd-to-trt

FROM nvcr.io/nvidia/l4t-base:r32.3.1

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

VOLUME /repo

RUN apt-get update && apt-get install -y python3-pip pkg-config zip

RUN python3 -m pip install --upgrade pip

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
RUN apt-get install -y python3-h5py
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow==1.15.2+nv20.2
RUN pip3 install wget pillow

COPY ./exporters/libflattenconcat.so.6 /opt/libflattenconcat.so 
COPY ./exporters/graphsurgeon.patch-4.2.2 /repo 
COPY install_trtexporter.sh /repo
RUN chmod +x /repo/install_trtexporter.sh && /repo/install_trtexporter.sh

# The `python3-opencv` package is old and doesn't support gstreamer video writer on Debian. So we need to manually build opencv.
ARG OPENCV_VERSION=4.3.0

# http://amritamaz.net/blog/opencv-config
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-vaapi \
        libavcodec-dev \
        libavformat-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libsm6 \
        libswscale-dev \
        libxext6 \
        libxrender-dev \
        mesa-va-drivers \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp/ \
    && curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -o opencv.tar.gz \
    && tar zxvf opencv.tar.gz && rm opencv.tar.gz \
    && cd /tmp/opencv-${OPENCV_VERSION} \
    && mkdir build \
    && cd build \
    && cmake \
        -DBUILD_opencv_python3=yes \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DINSTALL_TESTS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_DOCS=OFF \
        ../ \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && rm -rf opencv-${OPENCV_VERSION} \
    && apt-get purge -y \
        cmake \
        git \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libxrender-dev \
    && apt-get autoremove -y



WORKDIR /repo

