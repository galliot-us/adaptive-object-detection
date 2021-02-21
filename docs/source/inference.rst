Run Inference
=============

On any of the docker containers you can run sample inference to get an output video: ::

    python3 inference.py --device DEVICE --input_video INPUT_VIDEO --out_dir OUT_DIR \
                    [--model_path MODEL_PATH] [--label_map LABEL_MAP] [--threshold THRESHOLD]  [--input_width INPUT_WIDTH]\
                    [--input_height INPUT_HEIGHT] [--out_width OUT_WIDTH] [--out_height OUT_HEIGHT]

Where:

:code:`DEVICE` should be one of the :code:`x86`, :code:`edgetpu` or :code:`jetson`.

:code:`INPUT_VIDEO` is the path to the input video file.

:code:`OUT_DIR` is a directory in which the script will save the output video file.

:code:`MODEL_PATH` is the path to the model file or directory. For :code:`x86` devices, it should be a directory that contains the :code:`saved_model` directory. For :code:`edgetpu` it should be a compiled :code:`tflite` file, and for :code:`jetson` devices, it should be a :code:`TRT Engine` file.

:code:`label_map` is a :code:`pbtxt` file which contains a series of mappings that connects a set of class IDs with the corresponding class names. For example if your detector predict 3 as the object label, with the help of label_map.pbtxt you can map this label to corresponding class. If you pass the model_path you should pass this argument too. A sample to this file can be find in utils/mscoco_label_map.pbtxt . If you use our Adaptive Learning service the model label map is exist in output.zip file

:code:`threshold` is the detector's threshold to detect objects.

:code:`INPUT_WIDTH` and :code:`INPUT_HEIGHT` are the width and height of the input of the model.

:code:`OUT_WIDTH` and :code:`OUT_HEIGHT` are the resolutions of output video.


