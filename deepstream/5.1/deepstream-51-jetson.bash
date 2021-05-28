#!/bin/bash

if [ $# -eq 2 ]
then
    videoPath=$1
    labelPath=$2
fi
videoPath="file://$videoPath"
labelPath="$labelPath"


if [[ ! -z "${labelPath}" ]]; then
    python3 /repo/generate_labels_from_pbtxt.py $labelPath
    mv /repo/deepstream-data/labels.txt .
else
    wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi
labelPath="$PWD/labels.txt"

export CUDA_VER=10.2
cd ../libs/nvdsinfer_customparser/ && make
cd ../nvdsinfer_custom_impl_ssd/ && make
cd ../../5.1/


python3 deepstream_ssd_parser.py --input_video $videoPath --label_path $labelPath --out_dir out/ --inference_type 0 --config config_infer_primary_ssd_mobilenet.txt 


