#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -f [Path of frozen graph]"
    exit 1 # Exit script after printing help
}

while getopts "d:f:h:" opt
do
    case "$opt" in

        f ) frozen_path="$2" ;;
        h ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

cd ssd_mobilenet_v2_coco && mkdir 1
cp $frozen_path 1/model.graphdef

wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt

deepstream-app -c source1_primary_ssd_mobilenet_v2_coco.txt

