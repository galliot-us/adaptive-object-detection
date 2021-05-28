#!/bin/bash

cd ssd_mobilenet_trt_neuralet/
if [ $# -eq 2 ]
then
    videoPath=$1
    labelPath=$2
fi

videoPath="$videoPath"
labelPath="$labelPath"

if [ -f "$videoPath" ]; then
    cp $videoPath /repo/deepstream-data/input_vid.mp4
else
    echo "video file not exists in $videoPath."
    exit 1
fi

if [[ ! -z "${labelPath}" ]]; then
    python3 /repo/generate_labels_from_pbtxt.py $labelPath
    mv /repo/deepstream-data/labels.txt .
else
    wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi

export CUDA_VER=10.2
cd ../../libs/nvdsinfer_custom_impl_ssd/ && make
cd ../nvdsinfer_customparser/ && make
cd ../../5.0/ssd_mobilenet_trt_neuralet/

deepstream-app -c deepstream_app_config_ssd_mobilenet.txt 
