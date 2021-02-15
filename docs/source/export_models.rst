Export Models to Edge Devices
=============================

With the Neuralet Edge Object Detection module, you can easily export your trained model to Nvidia's Jetson Devices and Google Edge TPUs.

Compile tflite Models to Edge TPU Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In amd64 with connected USB Accelerator docker container run: ::

    python3 exporters/edgetpu_exporter.py --tflite_file TFLITE_FILE --out_dir OUT_DIR

Where :code:`TFLITE_FILE` should be a quantized model. You can use our Adaptive Learning API to train a quantized object detection model.

For more information about quantization techniques of deep neural networks, you can read our `blog <https://neuralet.com/article/quantization-of-tensorflow-object-detection-api-models/>`_.

Export TensorFlow protobuf models to TRT engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Jetson docker container run: ::

    python3 exporters/trt_exporter.py --pb_file PB_FILE --out_dir OUT_DIR [ --num_classes NUM_CLASSES]

Where :code:`PB_FILE` is a protobuf frozen graph tensorflow model.



