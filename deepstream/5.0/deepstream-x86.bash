#!/bin/bash


if [ $# -eq 2 ]
then
    videoPath=$1
    labelPath=$2
fi
cd ssd_mobilenet_v2_coco && mkdir -p 1
cp /repo/deepstream-data/frozen_inference_graph.pb 1/model.graphdef

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


deepstream-app -c source1_primary_ssd_mobilenet_v2_coco.txt

