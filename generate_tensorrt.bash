#!/bin/bash
# This script generates a TensorRT engine from pretrained Tensorflow SSD Mobilenet v2 COCO model.
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
python3 exporters/trt_exporter.py --pb_file ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --out_dir detectors/data/ --neuralet_adaptive_model 0
mv /repo/detectors/data/frozen_inference_graph.bin /repo/detectors/data/
