"""build_engine.py

This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.

Input : spces of a ssd frozen inference graph in config.ini file
Output: TensorRT Engine file

Reference:
   https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py
"""


import os
import ctypes
import argparse
import configparser
import wget

import uff
import tensorrt as trt
import graphsurgeon as gs
import numpy as np
import add_plugin_and_preprocess_ssd_mobilenet as plugin

def export_trt(pb_file, output_dir, num_classes=90):
    lib_flatten_concat_file = "exporters/libflattenconcat.so.6"
    # initialize
    if trt.__version__[0] < '7':
        ctypes.CDLL(lib_flatten_concat_file)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    
    # compile the model into TensorRT engine
    model = "ssd_mobilenet_v2_coco"

    if not os.path.isfile(pb_file):
        raise FileNotFoundError('model does not exist under: {}'.format(pb_file))

    dynamic_graph = plugin.add_plugin_and_preprocess(
        gs.DynamicGraph(pb_file),
        model,
        num_classes)
    model_file_name = ".".join((pb_file.split("/")[-1]).split(".")[:-1])
    uff_path = os.path.join(output_dir, model_file_name + ".uff")
    _ = uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        output_nodes=['NMS'],
        output_filename=uff_path,
        text=True,
        debug_mode=False)
    input_dims = (3, 300, 300)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', input_dims)
        parser.register_output('MarkOutput_0')
        parser.parse(uff_path, network)
        engine = builder.build_cuda_engine(network)
        
        buf = engine.serialize()
        engine_path = os.path.join(output_dir, model_file_name + ".bin")
        with open(engine_path, 'wb') as f:
            f.write(buf)
        print("your model has been converted to trt engine successfully under : {}".format(engine_path))

