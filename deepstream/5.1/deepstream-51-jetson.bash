#!/bin/bash

if [ $# -eq 1 ]
then
    videoPath=$1
fi

videoPath="file://$videoPath"

cd ssd_mobilenet_v2 && mkdir -p 1
cp /repo/deepstream-data/frozen_inference_graph.pb 1/model.graphdef

label_file_path="/repo/deepstream/5.1/ssd_mobilenet_v2/labels.txt"
if [ ! -f $label_file_path ]; then
        wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi

cd ..
python3 deepstream_ssd_parser.py $videoPath $label_file_path

