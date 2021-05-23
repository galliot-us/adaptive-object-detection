#!/bin/bash

if [ $# -eq 1 ]
then
    videoPath=$1
fi

videoPath="file://$videoPath"
labelPath="$labelPath"

cd ssd_mobilenet_v2 && mkdir -p 1
if [ ! -f /repo/deepstream-data/frozen_inference_graph.pb ]; then
    echo "Couldn't find frozen_inference graph, downloading from model zoo...!"
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    cp ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb 1/model.graphdef
    rm -rf ssd_mobilenet_v2_coco_2018_03_29.tar.gz ssd_mobilenet_v2_coco_2018_03_29/
else
    cp /repo/deepstream-data/frozen_inference_graph.pb 1/model.graphdef
fi


if [[ ! -z "${labelPath}" ]]; then
    bash /repo/generate_label.py $labelPath
    mv /repo/deepstream-data/labels.txt .
else
    wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_triton_model_deploy/master/faster_rcnn_inception_v2/config/labels.txt
fi

labelPath="$PWD/labels.txt"

cd ..
python3 deepstream_ssd_parser.py --input_video $videoPath --label_path $label_file_path --out_dir out/ --inference_type 1 --config dstest_ssd_nopostprocess.txt
