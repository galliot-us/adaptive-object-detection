#!/bin/bash

cd ssd_mobilenet_v2_coco && mkdir -p 1
cp /repo/deepstream-data/frozen_inference_graph.pb 1/model.graphdef

label_file_path="/repo/ssd_mobilenet_v2_coco/labels.txt"
if [ ! -f $label_file_path ]; then
    wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi

deepstream-app -c source1_primary_ssd_mobilenet_v2_coco.txt

