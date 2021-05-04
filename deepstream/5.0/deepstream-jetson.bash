#!/bin/bash

cd ssd_mobilenet_trt_neuralet/

label_file_path="/repo/deepstream/5.0/ssd_mobilenet_trt_neuralet/labels.txt"
pwd && ls $label_file_path
if [ ! -f $label_file_path ]; then
    wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi

export CUDA_VER=10.2
cd nvdsinfer_custom_impl_ssd/ && make
cd ../nvdsinfer_customparser/ && make
cd ../ 
deepstream-app -c deepstream_app_config_ssd_mobilenet.txt 
