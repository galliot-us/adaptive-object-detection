Deploy Model Using NVIDIA DeepStream
====================================

After downloading your trained student :code:`ssd_mobilenet_v2` model, you can deploy it by using DeepStream or if you want to run inference on the pretrained model dfrom the Tensorflow Model Zoo you can download it from `here <http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz>`_.

You need to copy your :code:`frozen_inference_graph` which is located in this
`train_outputs/frozen_graph/frozen_inference_graph.pb`
path to deepstream-data directory. ::

    cp train_outputs/frozen_graph/frozen_inference_graph.pb deepstream-data/ 

Then you need to generate and put a `label.txt` file to the :code:`ssd_mobilenet_v2_coco` directory. If you have trained your model with all of the coco labels, you can skip this step as the label file will be downloaded automatically. The label.txt file consists of trained model labels, each in one line without any sign or numbeir.

:code:`source1_primary_ssd_mobilenet_v2_coco.txt` is the DeepStream main configuration file and consists of properties of components in the Deepstream pipeline. You need to set your video path in :code:`uri` key of the :code:`source0` section and your output video path in :code:`output-file` property in :code:`sink0` section.

:code:`Config_infer_primary_ssd_mobilenet_v2_coco.txt` is the Deepstream inference configuration file and consists of the inference plugin information regarding the inputs, outputs, preprocessing, post-processing, and communication facilities required by the application. If you need to change :code:`label.txt` path, you can edit :code:`labelfile_path` in :code:`postprocess` section in this file.

After setting all above-mentioned properties, you can use the following commands to run Neuralet docker container for running inference using the DeepStream: ::

    docker build -f deepstream-x86.Dockerfile -t "neuralet/object-detection:deepstream-x86" .
    docker run -it --gpus all  --runtime nvidia -v "$PWD":/repo  neuralet/object-detection:deepstream-x86

Run on Jetson Devices
^^^^^^^^^^^^^^^^^^^^^

First you need to generate TensorRT using :code:`exporter/trt_exporter.py` script. It will generate a :code:`frozen_inference_graph.bin` engine file and you need to copy it inside deepstream data directory.
Then you need to generate and put the `label.txt` file to the :code:`ssd_mobilenet_trt_neuralet` directory. 

You can set your video path in :code:`uri` key of the :code:`source0 section and your output video path in :code:`output-file` property in :code:`sink0` section in :code:`deepstream_app_config_ssd_mobilenet.txt`.

After setting all above-mentioned properties, you can use the following commands to run Neuralet docker container for running inference using the DeepStream: ::

    docker build -f deepstream-jetson.Dockerfile -t "neuralet/object-detection:deepstream-jetson-4-4" .
    docker run --runtime nvidia --privileged -it -v $PWD:/repo neuralet/object-detection:deepstream-jetson-4-4
 
